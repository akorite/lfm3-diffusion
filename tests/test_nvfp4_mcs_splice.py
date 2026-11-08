"""CPU synthetic test for the MCS-NVFP4 splice + input_scale recalibration.

Splicing trained BF16 mixer/norm/embedding tensors onto an NVFP4-shaped
checkpoint must preserve `U8 + F8_E4M3` MLP weight checksums bit-identically
while updating every F32 `input_scale` scalar to `amax / 2688.0` (x headroom,
>= 1.0 so nothing clips).
"""
import torch

from scripts.export_fp8_nvfp4 import (
    NVFP4_AMAX_DENOMINATOR,
    splice_nvfp4_state_dict,
    update_input_scales,
)

NUM_LAYERS = 4  # synthetic stands in for 24 layers / 72 input_scales
HIDDEN = 32
INTER = 48
MLP_LINEARS = ("w1", "w3", "w2")


def _nvfp4_base() -> dict[str, torch.Tensor]:
    """Mock `LiquidAI/LFM3-1.7B-NVFP4` layout: U8 packed FP4 weights, F8_E4M3
    weight scales, F32 input_scale per MLP linear; BF16 everything else."""
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(100, HIDDEN, dtype=torch.bfloat16),
    }
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for w in MLP_LINEARS:
            rows, cols = (INTER, HIDDEN) if w in ("w1", "w3") else (HIDDEN, INTER)
            sd[f"{p}.feed_forward.{w}.weight_packed"] = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8
            )
            sd[f"{p}.feed_forward.{w}.weight_scale"] = torch.rand(
                rows, cols // 16, dtype=torch.float32
            ).to(torch.float8_e4m3fn)
            sd[f"{p}.feed_forward.{w}.input_scale"] = torch.tensor(1.0)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16)
        sd[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    return sd


def _trained() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(100, HIDDEN, dtype=torch.bfloat16),
    }
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16)
        sd[f"{p}.input_layernorm.weight"] = torch.full((HIDDEN,), 2.0, dtype=torch.bfloat16)
        # Trained BF16 MLP weights must NOT clobber the NVFP4 base.
        for w in MLP_LINEARS:
            sd[f"{p}.feed_forward.{w}.weight"] = torch.randn(4, 4, dtype=torch.bfloat16)
    sd["model.layers.9.self_attn.q_proj.weight"] = torch.randn(2, 2)  # aux key, dropped
    return sd


def test_splice_preserves_mlp_weights_and_replaces_the_rest():
    base, trained = _nvfp4_base(), _trained()
    merged = splice_nvfp4_state_dict(base, trained)

    for k, v in base.items():
        if ".feed_forward." in k:
            # fp8 tensors lack CPU `equal`; compare raw bytes.
            lhs, rhs = merged[k], v
            if lhs.dtype.is_floating_point and lhs.dtype.itemsize == 1:
                lhs, rhs = lhs.view(torch.uint8), rhs.view(torch.uint8)
            assert torch.equal(lhs, rhs), f"MLP key {k} was mutated"

    n_scales = sum(k.endswith("input_scale") for k in merged)
    assert n_scales == NUM_LAYERS * len(MLP_LINEARS)

    for k, v in trained.items():
        if ".feed_forward." in k or k not in base:
            continue
        assert torch.equal(merged[k], v), f"trained key {k} not spliced"

    assert "model.layers.9.self_attn.q_proj.weight" not in merged


def test_input_scales_update_to_amax_over_2688():
    merged = splice_nvfp4_state_dict(_nvfp4_base(), _trained())
    torch.manual_seed(2)
    amax = {}
    for i in range(NUM_LAYERS):
        for w in MLP_LINEARS:
            amax[f"model.layers.{i}.feed_forward.{w}"] = float(torch.rand(()) * 2000 + 100)

    updated = update_input_scales(merged, amax)

    for prefix, a in amax.items():
        got = updated[prefix + ".input_scale"].item()
        expected = torch.tensor(a / NVFP4_AMAX_DENOMINATOR, dtype=torch.float32).item()
        assert got == expected  # bit-exact float32 division, no clipping

    # Non-scale tensors untouched.
    for k, v in merged.items():
        if not k.endswith(".input_scale"):
            assert _bitwise_equal(updated[k], v)

    # Headroom keeps amax strictly inside the scaled range (no clipping).
    padded = update_input_scales(merged, amax, headroom=1.1)
    for prefix, a in amax.items():
        assert padded[prefix + ".input_scale"].item() * NVFP4_AMAX_DENOMINATOR > a


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.equal that also works on fp8 dtypes (no CPU equal kernel)."""
    if a.dtype.is_floating_point and a.dtype.itemsize == 1:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a, b)


def test_update_scales_rejects_headroom_below_one_and_missing_modules():
    merged = splice_nvfp4_state_dict(_nvfp4_base(), _trained())
    try:
        update_input_scales(merged, {"model.layers.0.feed_forward.w1": 5.0}, headroom=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("headroom < 1.0 must raise")
    try:
        update_input_scales(merged, {"model.layers.0.feed_forward.missing": 5.0})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown module prefix must raise")
