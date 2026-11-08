"""Static-FP8 frozen SwiGLU MLP wrapper.

`w1` + `w3` are fused into `w13 [2*intermediate, hidden]` once; `w13` and
`w2` are stored (with contiguous transposes for the dgrad GEMM) and run
through a custom `autograd.Function` that computes forward via
`torch._scaled_mm` rowwise FP8 on CUDA and dgrad only when
`ctx.needs_input_grad[0]` — frozen weights need no wgrad. CPU (and any
device without `_scaled_mm`) falls back to bf16 matmul. Weights carry
`requires_grad=False`, so backward is dgrad-only either way.

`torchao`'s Float8Linear is intentionally not used here: it computes
wgrad unconditionally, which defeats the frozen-MLP FLOP saving.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer import MASK_TOKEN_ID


def _scaled_mm_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch, "_scaled_mm")


class _FrozenMatMul(torch.autograd.Function):
    """y = x @ W^T with dgrad-only backward (bf16 weights)."""

    @staticmethod
    def forward(ctx, x, w, w_t):
        if w.requires_grad or w_t.requires_grad:
            ctx.save_for_backward(x, w_t)
            ctx.has_x = True
        else:
            ctx.save_for_backward(w_t)
            ctx.has_x = False
        return x.to(torch.bfloat16) @ w.to(torch.bfloat16).t()

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        x = saved[0] if ctx.has_x else None
        w_t = saved[1] if ctx.has_x else saved[0]
        dx = None
        if ctx.needs_input_grad[0]:
            dx = dy.to(torch.bfloat16) @ w_t.to(torch.bfloat16).t()
        dw = dw_t = None
        if x is not None and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            d = (dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16).t()
                 @ x.reshape(-1, x.shape[-1]).to(torch.bfloat16))
            if ctx.needs_input_grad[1]:
                dw = d
            if ctx.needs_input_grad[2]:
                dw_t = d.t()
        return dx, dw, dw_t


def _quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel e4m3 quantization: returns `(q [N,H], scale [1,N])`."""
    scale = (w.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    q = (w.float() / scale).to(torch.float8_e4m3fn)
    return q, scale.t().contiguous()


class _Fp8MatMul(torch.autograd.Function):
    """Rowwise-dynamic-activation, per-channel-weight `_scaled_mm` GEMM.

    The activation scale is computed per forward (dynamic), the weight
    scale is fixed at freeze time, and `dgrad` runs against the bf16 copy
    the frozen weights keep for exactly this purpose.
    """

    @staticmethod
    def forward(ctx, x, w8, w_scale, w_bf16):
        ctx.save_for_backward(w_bf16)
        flat = x.reshape(-1, x.shape[-1]).float()
        a_scale = (flat.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0).contiguous()
        a8 = (flat / a_scale).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(a8, w8.t(), scale_a=a_scale, scale_b=w_scale,
                             out_dtype=torch.bfloat16)
        return y.view(*x.shape[:-1], w8.shape[0])

    @staticmethod
    def backward(ctx, dy):
        (w_bf16,) = ctx.saved_tensors
        dx = dy.to(torch.bfloat16) @ w_bf16 if ctx.needs_input_grad[0] else None
        return dx, None, None, None


class FrozenMLP(nn.Module):
    """Fused frozen SwiGLU: `w2 @ (silu(a) * b)` where `[a;b] = w13 @ x`."""

    def __init__(self, mlp: nn.Module, fp8: bool = False):
        super().__init__()
        w13 = torch.cat([mlp.w1.weight, mlp.w3.weight], 0).contiguous()
        w2 = mlp.w2.weight.contiguous()
        self.intermediate = mlp.w1.out_features
        self.fp8 = fp8 and _scaled_mm_available()
        if self.fp8:
            # weights are frozen, so the quantization is done once; the bf16
            # copy exists only for the input gradient
            q13, s13 = _quantize_rows(w13)
            q2, s2 = _quantize_rows(w2)
            self.w13 = nn.Parameter(q13, requires_grad=False)
            self.w2 = nn.Parameter(q2, requires_grad=False)
            self.w13_scale = nn.Parameter(s13, requires_grad=False)
            self.w2_scale = nn.Parameter(s2, requires_grad=False)
            self.w13_bf16 = nn.Parameter(w13.to(torch.bfloat16), requires_grad=False)
            self.w2_bf16 = nn.Parameter(w2.to(torch.bfloat16), requires_grad=False)
        else:
            self.w13 = nn.Parameter(w13.to(torch.bfloat16), requires_grad=False)
            self.w2 = nn.Parameter(w2.to(torch.bfloat16), requires_grad=False)
            self.w13_cm = nn.Parameter(w13.t().contiguous().to(torch.bfloat16), requires_grad=False)
            self.w2_cm = nn.Parameter(w2.t().contiguous().to(torch.bfloat16), requires_grad=False)

    def _linear(self, x, w, w_t, w_scale=None, w_bf16=None):
        if self.fp8 and not torch.is_inference_mode_enabled():
            return _Fp8MatMul.apply(x, w, w_scale, w_bf16)
        if torch.is_inference_mode_enabled():
            return x.to(torch.bfloat16) @ w.to(torch.bfloat16).t()
        return _FrozenMatMul.apply(x, w, w_t)

    def forward(self, x):
        if self.fp8:
            y13 = self._linear(x, self.w13, None, self.w13_scale, self.w13_bf16)
        else:
            y13 = self._linear(x, self.w13, self.w13_cm)
        a, b = y13.chunk(2, -1)
        h = F.silu(a) * b
        if self.fp8:
            return self._linear(h, self.w2, None, self.w2_scale, self.w2_bf16)
        return self._linear(h, self.w2, self.w2_cm)


def freeze_mlp_weights(model: nn.Module, fp8: bool = None) -> int:
    """Replace every `feed_forward` Lfm3MLP with `FrozenMLP` and freeze all
    vocabulary rows of the tied embedding except the mask row, which is the
    only new output the head has to learn. `fp8` defaults to `_scaled_mm`
    availability (off on CPU). Returns the number of swapped MLPs."""
    if fp8 is None:
        fp8 = False  # opt-in: the rowwise FP8 path is validated on hardware first
    n = 0
    for layer in model.model.layers:
        if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "w1"):
            layer.feed_forward = FrozenMLP(layer.feed_forward, fp8=fp8)
            n += 1
    emb = model.model.embed_tokens
    emb.weight.requires_grad_(True)  # mask row must learn
    keep = torch.zeros(emb.weight.shape[0], 1, dtype=emb.weight.dtype, device=emb.weight.device)
    keep[MASK_TOKEN_ID] = 1.0
    emb.weight.register_hook(lambda g: g.mul_(keep))
    return n


def unfreeze_mlp_weights(model: nn.Module) -> int:
    """KTD2 fallback gate: set MLP params `requires_grad=True` for the
    `LR=1e-6` polish. Returns count of unfrozen MLPs."""
    n = 0
    for layer in model.model.layers:
        if isinstance(layer.feed_forward, FrozenMLP):
            if layer.feed_forward.fp8:
                raise ValueError(
                    "cannot unfreeze quantized MLP weights: re-freeze with "
                    "fp8=False before running the polish"
                )
            for p in layer.feed_forward.parameters():
                p.requires_grad_(True)
            n += 1
    return n
