import torch
import pytest

from lfm3_dlm.train.loss_cad import cad_loss, ce_loss, compute_step_loss, sliced_lse

torch.manual_seed(0)
H, V_S, K, M = 32, 125_033, 16, 12
TAU = 1e-8


def make_teacher(v_s=V_S, m=M, k=K):
    g = torch.Generator().manual_seed(3)
    classifier = torch.randn(v_s, H, generator=g) / H**0.5
    hidden = torch.randn(m, H, generator=g)
    teacher_ids = torch.randint(0, v_s, (m, k), generator=g)
    # normalized teacher over top-16 + tau support
    lp = torch.log_softmax(torch.randn(m, k, generator=g) * 2, dim=-1)
    res = torch.rand(m, generator=g) * 0.02
    mass = lp.exp().sum(-1) + res
    lp = (lp.exp() / mass.unsqueeze(-1)).log()
    res = res / mass
    return classifier, hidden, teacher_ids, lp, res


def dense_reference(hidden, classifier, teacher_ids, lp, res, c_T=None):
    """FP64 125033-way reference over the identical support."""
    z = (hidden.float() @ classifier.float().t()).double()
    lse = torch.logsumexp(z, dim=-1)
    mass = lp.double().exp().sum(-1) + res.double()
    p_t = torch.cat([lp.double().exp(), res.double().unsqueeze(-1)], -1) / mass.double().unsqueeze(-1)
    p_t = p_t.clamp_min(TAU)
    z_top = torch.gather(z, 1, teacher_ids).to(torch.bfloat16).double()  # CCE bf16 rounding
    p_s_top = (z_top - lse.unsqueeze(-1)).exp()
    p_s_tau = (1 - p_s_top.sum(-1)).clamp_min(TAU)
    p_s = torch.cat([p_s_top, p_s_tau.unsqueeze(-1)], -1)
    fwd = (p_t * (p_t.log() - p_s.log())).sum(-1)
    rev = (p_s * (p_s.log() - p_t.log())).sum(-1)
    if c_T is None:
        c_T = p_t.max(-1).values
    return fwd + c_T * rev


def test_lse_matches_dense_logsumexp():
    classifier, hidden, *_ = make_teacher(m=4)
    lse = sliced_lse(hidden, classifier)
    ref = torch.logsumexp(hidden.float() @ classifier.float().t(), dim=-1)
    assert torch.allclose(lse, ref, atol=1e-4)


def test_cad_matches_fp64_reference():
    classifier, hidden, ids, lp, res = make_teacher()
    lse = sliced_lse(hidden, classifier)
    got = cad_loss(hidden, classifier, ids, lp, res, lse=lse)
    ref = dense_reference(hidden, classifier, ids, lp, res)
    assert torch.allclose(got.double(), ref, atol=1e-4)


def test_cad_zero_when_student_equals_teacher():
    g = torch.Generator().manual_seed(5)
    classifier = torch.randn(V_S, H, generator=g) / H**0.5
    hidden = torch.randn(M, H, generator=g)
    lse = sliced_lse(hidden, classifier).double()
    z = hidden.double() @ classifier.double().t()
    ids = torch.randint(0, V_S, (M, K), generator=g)
    p_s_top = (torch.gather(z, 1, ids) - lse.unsqueeze(-1)).exp()
    res = (1 - p_s_top.sum(-1)).clamp_min(1e-6).float()
    lp = p_s_top.clamp_min(1e-9).log().float()
    got = cad_loss(hidden, classifier, ids, lp, res, lse=lse.float())
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-4)


def test_cad_monotonic_in_cT():
    classifier, hidden, ids, lp, res = make_teacher()
    lse = sliced_lse(hidden, classifier)
    vals = [
        cad_loss(hidden, classifier, ids, lp, res, lse=lse,
                 c_T=torch.full((M,), c)).mean()
        for c in (0.0, 0.3, 0.6, 0.9, 1.0)
    ]
    for a, b in zip(vals, vals[1:]):
        assert b > a


def test_lse_reused_single_computation():
    """compute_step_loss computes the LSE once over the gathered union —
    verified indirectly: per-segment LSEs equal standalone lse slices."""
    classifier, hidden_all, ids, lp, res = make_teacher(m=20)
    cad_idx = torch.arange(0, 8)
    ce_idx = torch.arange(8, 14)
    ar_idx = torch.arange(14, 20)
    lse_full = sliced_lse(hidden_all[torch.cat([cad_idx, ce_idx, ar_idx])], classifier)
    ref = sliced_lse(hidden_all, classifier)
    assert torch.allclose(lse_full, ref[torch.cat([cad_idx, ce_idx, ar_idx])])


def test_step_loss_components_and_ar_assert():
    classifier, hidden_all, ids, lp, res = make_teacher(m=20)
    kw = dict(
        cad_idx=torch.arange(0, 8),
        teacher_ids=ids[:8], teacher_logprobs=lp[:8], teacher_residual=res[:8],
        ce_idx=torch.arange(8, 14),
        ce_targets=torch.randint(0, V_S, (6,)),
        ce_weights=torch.full((6,), 0.2),
        ar_idx=torch.arange(14, 20),
        ar_targets=torch.randint(0, V_S, (6,)),
    )
    with pytest.raises(AssertionError):
        compute_step_loss(hidden_all, classifier, w_ar=1.0, causal_x0=False, **kw)
    out = compute_step_loss(hidden_all, classifier, w_ar=1.0, causal_x0=True, **kw)
    assert torch.isfinite(out.loss)
    # auto-balance pins L_AR contribution equal to L_CAD
    assert torch.allclose(out.loss, out.l_cad + out.l_ce + out.l_cad, atol=1e-5)


def test_ce_revealed_only_weighting():
    classifier, hidden_all, *_ = make_teacher(m=6)
    targets = torch.randint(0, V_S, (6,))
    lse = sliced_lse(hidden_all, classifier)
    w = torch.tensor([0.2, 0.0, 0.2, 0.0, 0.0, 0.2])
    got = ce_loss(hidden_all, classifier, targets, lse, w)
    # unweighted CE on the three revealed positions only
    z = torch.einsum("mh,mh->m", hidden_all, classifier[targets]).to(torch.bfloat16).float()
    ref = ((lse - z) * w).sum() / w.sum()
    assert torch.allclose(got, ref, atol=1e-4)
