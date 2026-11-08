"""Reused-LSE `CForce` CAD loss + 0.2*CE + auto-balanced `L_AR`.

The output softmax is computed over the *sliced* classifier
`lm_head.weight[:OUTPUT_VOCAB_BOUNDARY]` so rows `125033..127999` never
enter the LSE. The LSE is computed once (via `cut-cross-entropy`
`linear_cross_entropy(..., return_lse=True, filter_eps=None)` on GPU, or
a pure-PyTorch reference on CPU) and reused for the Top-16 forward /
reverse KL, the per-position CE, and the clean-stream `L_AR`.

CForce CAD over teacher Top-16 + one tail bucket tau:
    p_T = [softmax(top-16 logprobs) | p_T(tau) renormalized]
    p_S = [exp(z_v - lse) | p_S(tau) = 1 - sum p_S(v)]
    L_CAD = KL(p_T || p_S) + c_T * KL(p_S || p_T),
    c_T = max_k p_T(k) over top-16 ∪ {tau}
Gather-dot logits are cast to bf16 before subtracting the LSE so the
rounding matches the fused CE path.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..modeling_lfm3 import OUTPUT_VOCAB_BOUNDARY

TAU_FLOOR = 1e-8
AR_BALANCE_EPS = 1e-8
# rows per classifier GEMM chunk: bounds the transient [rows, V] fp32 block
LSE_CHUNK = 512


def sliced_lse(hidden: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Exact log-sum-exp over sliced classifier rows, computed once.

    `hidden` [M, H] float, `classifier` [V_s, H] (already sliced). Returns
    `lse` [M] float32. Uses cut-cross-entropy when available (CUDA);
    falls back to a single fp32 matmul on CPU.
    """
    try:
        from cut_cross_entropy import linear_cross_entropy  # type: ignore

        dummy_targets = torch.zeros(
            hidden.shape[0], dtype=torch.long, device=hidden.device
        )
        _, lse = linear_cross_entropy(
            hidden.to(torch.bfloat16),
            classifier.to(torch.bfloat16),
            dummy_targets,
            return_lse=True,
            filter_eps=None,
        )
        return lse.float()
    except ImportError:
        return chunked_lse(hidden, classifier)


def chunked_lse(hidden: torch.Tensor, classifier: torch.Tensor,
                chunk: int = LSE_CHUNK) -> torch.Tensor:
    """Log-sum-exp over the sliced classifier in row chunks.

    Allocating `[positions, 125033]` as one fp32 block costs 3-6 GB on a packed
    row; chunking keeps the transient bounded and the result identical.
    """
    h = hidden.float()
    w = classifier.float().t()
    out = []
    for start in range(0, h.shape[0], chunk):
        out.append(torch.logsumexp(h[start:start + chunk] @ w, dim=-1))
    return torch.cat(out) if out else h.new_zeros(0)


def gather_logits(
    hidden: torch.Tensor, classifier: torch.Tensor, vocab_ids: torch.Tensor
) -> torch.Tensor:
    """`z[v] = hidden . classifier[v]` for candidate `vocab_ids` [M, K],
    cast to bf16 (CCE rounding) and returned as float32."""
    w = classifier[vocab_ids]  # [M, K, H]
    z = torch.einsum("mh,mkh->mk", hidden, w).to(torch.bfloat16)
    return z.float()


def cad_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_residual: torch.Tensor,
    lse: torch.Tensor | None = None,
    c_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """CForce confidence-adaptive KL per position, [M].

    Teacher distribution is renormalized over its top-16 ∪ {tau} support
    so `L_CAD == 0` exactly when p_theta = p_T.
    """
    if lse is None:
        lse = sliced_lse(hidden, classifier)

    # teacher distribution over its own support
    t_res = teacher_residual.clamp_min(TAU_FLOOR)
    mass = teacher_logprobs.exp().sum(-1) + t_res
    p_t = torch.cat([teacher_logprobs.exp(), t_res.unsqueeze(-1)], -1) / mass.unsqueeze(-1)
    p_t = p_t.clamp_min(TAU_FLOOR)
    log_p_t = p_t.log()

    # student distribution over the same support
    z = gather_logits(hidden, classifier, teacher_ids)  # [M, 16]
    log_p_s_top = z - lse.unsqueeze(-1)
    p_s_top = log_p_s_top.exp()
    p_s_tau = (1.0 - p_s_top.sum(-1)).clamp_min(TAU_FLOOR)
    p_s = torch.cat([p_s_top, p_s_tau.unsqueeze(-1)], -1)
    log_p_s = p_s.log()

    fwd = (p_t * (log_p_t - log_p_s)).sum(-1)
    rev = (p_s * (log_p_s - log_p_t)).sum(-1)
    if c_T is None:
        c_T = p_t.max(-1).values
    return fwd + c_T * rev


def ce_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    lse: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted cross-entropy reusing the precomputed LSE. Returns scalar."""
    if hidden.shape[0] == 0:
        return hidden.new_zeros((), dtype=torch.float32)
    z = gather_logits(hidden, classifier, targets.unsqueeze(-1)).squeeze(-1)
    ce = lse - z
    if weight is None:
        return ce.mean()
    wsum = weight.sum().clamp_min(AR_BALANCE_EPS)
    return (ce * weight).sum() / wsum


@dataclass
class CADLossOutput:
    loss: torch.Tensor
    l_cad: torch.Tensor
    l_ce: torch.Tensor
    l_ar: torch.Tensor
    c_T_mean: torch.Tensor


def compute_step_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    *,
    cad_idx: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_residual: torch.Tensor,
    ce_idx: torch.Tensor,
    ce_targets: torch.Tensor,
    ce_weights: torch.Tensor,
    ar_idx: torch.Tensor,
    ar_targets: torch.Tensor,
    w_ar: float = 1.0,
    causal_x0: bool = False,
) -> CADLossOutput:
    """Full single-stage step: L_CAD + 0.2*CE + auto-balanced L_AR.

    `hidden` [S, H] full-row hidden states; `*_idx` index tensors gather
    the active positions before the classifier (the head FLOP saving).
    When `w_ar > 0` the caller must assert the clean stream is causal
    (`causal_x0`), since L_AR is only meaningful on x0.
    """
    if w_ar > 0:
        assert causal_x0, "L_AR requires causal clean-stream hidden states"

    idx = torch.cat([cad_idx, ce_idx, ar_idx])
    lse = sliced_lse(hidden[idx], classifier) if idx.numel() else hidden.new_zeros(0)
    n_cad, n_ce = cad_idx.numel(), ce_idx.numel()
    lse_cad, lse_ce, lse_ar = lse[:n_cad], lse[n_cad : n_cad + n_ce], lse[n_cad + n_ce :]

    if n_cad:
        p_t_support = torch.cat(
            [teacher_logprobs.exp(), teacher_residual.clamp_min(TAU_FLOOR).unsqueeze(-1)], -1
        )
        p_t_support = p_t_support / p_t_support.sum(-1, keepdim=True)
        c_T = p_t_support.max(-1).values
        l_cad = cad_loss(
            hidden[cad_idx], classifier,
            teacher_ids, teacher_logprobs, teacher_residual, lse=lse_cad, c_T=c_T,
        ).mean()
        c_T_mean = c_T.mean().detach()
    else:
        l_cad = hidden.new_zeros(())
        c_T_mean = l_cad

    l_ce = ce_loss(hidden[ce_idx], classifier, ce_targets, lse_ce, ce_weights)

    if w_ar > 0 and ar_idx.numel():
        l_ar = ce_loss(hidden[ar_idx], classifier, ar_targets, lse_ar)
        # with no CAD positions to balance against, the AR term stands alone
        balanced = (l_ar if not n_cad
                    else l_cad.detach() / l_ar.detach().clamp_min(AR_BALANCE_EPS) * l_ar)
        loss = l_cad + l_ce + w_ar * balanced
    else:
        l_ar = hidden.new_zeros(())
        loss = l_cad + l_ce

    return CADLossOutput(
        loss=loss,
        l_cad=l_cad.detach(),
        l_ce=l_ce.detach(),
        l_ar=l_ar.detach(),
        c_T_mean=c_T_mean,
    )


@torch.no_grad()
def oput_corrupt(hidden: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Detached student top-1 over the sliced classifier: the on-manifold
    corruption source for the phase-2 OPUT steps."""
    return (hidden.float() @ classifier.float().t()).argmax(-1)
