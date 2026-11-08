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


def _scaled_mm_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch, "_scaled_mm")


class _FrozenMatMul(torch.autograd.Function):
    """y = x @ W^T with dgrad-only backward.

    Forward: `torch._scaled_mm` (rowwise e4m3) when available else a bf16
    matmul. Backward computes `dx = dy @ W` only when the input grad is
    actually requested.
    """

    @staticmethod
    def forward(ctx, x, w, w_t):
        ctx.save_for_backward(w_t)
        ctx.x_needs_grad = x.requires_grad
        if _scaled_mm_available() and w.dtype == torch.float8_e4m3fn:
            sx = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)
            sw = torch.ones(w.shape[0], 1, device=x.device, dtype=torch.float32)
            return torch._scaled_mm(
                x.reshape(-1, x.shape[-1]),
                w.t(),
                scale_a=sx,
                scale_b=sw,
                out_dtype=torch.bfloat16,
            ).view(*x.shape[:-1], w.shape[0])
        return x.to(torch.bfloat16) @ w.to(torch.bfloat16).t()

    @staticmethod
    def backward(ctx, dy):
        (w_t,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            if _scaled_mm_available() and w_t.dtype == torch.float8_e4m3fn:
                sx = torch.ones(dy.numel() // dy.shape[-1], 1, device=dy.device, dtype=torch.float32)
                sw = torch.ones(w_t.shape[0], 1, device=dy.device, dtype=torch.float32)
                dx = torch._scaled_mm(
                    dy.reshape(-1, dy.shape[-1]),
                    w_t.t(),
                    scale_a=sx,
                    scale_b=sw,
                    out_dtype=torch.bfloat16,
                ).view(*dy.shape[:-1], w_t.shape[0])
            else:
                dx = dy.to(torch.bfloat16) @ w_t.to(torch.bfloat16).t()
        return dx, None, None


class FrozenMLP(nn.Module):
    """Fused frozen SwiGLU: `w2 @ (silu(a) * b)` where `[a;b] = w13 @ x`."""

    def __init__(self, mlp: nn.Module, fp8: bool = False):
        super().__init__()
        w13 = torch.cat([mlp.w1.weight, mlp.w3.weight], 0).contiguous()
        w2 = mlp.w2.weight.contiguous()
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        self.intermediate = mlp.w1.out_features
        self.w13 = nn.Parameter(w13.to(dtype), requires_grad=False)
        self.w2 = nn.Parameter(w2.to(dtype), requires_grad=False)
        self.w13_cm = nn.Parameter(w13.t().contiguous().to(dtype), requires_grad=False)
        self.w2_cm = nn.Parameter(w2.t().contiguous().to(dtype), requires_grad=False)

    def forward(self, x):
        if torch.is_inference_mode_enabled():
            # FP8 params trip `version_counter` under inference_mode on
            # some torchao paths; run through a no_grad math fallback.
            with torch.no_grad():
                y13 = x.to(torch.bfloat16) @ self.w13.to(torch.bfloat16).t()
                a, b = y13.chunk(2, -1)
                return (F.silu(a) * b) @ self.w2.to(torch.bfloat16).t()
        y13 = _FrozenMatMul.apply(x, self.w13, self.w13_cm)
        a, b = y13.chunk(2, -1)
        h = F.silu(a) * b
        return _FrozenMatMul.apply(h, self.w2, self.w2_cm)


def freeze_mlp_weights(model: nn.Module, fp8: bool = None) -> int:
    """Replace every `feed_forward` Lfm3MLP with `FrozenMLP` and freeze the
    active embedding rows (mask row 125033 stays trainable). `fp8`
    defaults to `_scaled_mm` availability (off on CPU). Returns the number
    of swapped MLPs."""
    if fp8 is None:
        fp8 = _scaled_mm_available()
    n = 0
    for layer in model.model.layers:
        if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "w1"):
            layer.feed_forward = FrozenMLP(layer.feed_forward, fp8=fp8)
            n += 1
    emb = model.model.embed_tokens
    emb.weight.requires_grad_(True)  # mask row must learn
    return n


def unfreeze_mlp_weights(model: nn.Module) -> int:
    """KTD2 fallback gate: set MLP params `requires_grad=True` for the
    `LR=1e-6` polish. Returns count of unfrozen MLPs."""
    n = 0
    for layer in model.model.layers:
        if isinstance(layer.feed_forward, FrozenMLP):
            for p in layer.feed_forward.parameters():
                p.requires_grad_(True)
            n += 1
    return n
