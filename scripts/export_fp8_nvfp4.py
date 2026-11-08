#!/usr/bin/env python3
"""Tri-format export driver: BF16, PerRow-FP8, and MCS-NVFP4 checkpoints.

Subcommands:
  bf16   Standalone BF16 export (delegates to lfm3_dlm.convert).
  fp8    `compressed-tensors` FP8_DYNAMIC recipe: 1D PerChannel W8 +
         dynamic per-row A8 on `feed_forward.{w1,w2,w3}` and
         `self_attn.{q,k,v,o}_proj`. Conv/gate_proj, embed/lm_head and
         norms stay BF16. If `llmcompressor` is importable the recipe is
         applied in-place; otherwise `fp8_recipe.json` is written for the
         GPU runbook step to apply.
  nvfp4  Splice trained BF16 mixer/norm/embedding tensors onto
         `LiquidAI/LFM3-1.7B-NVFP4`'s U8 + F8_E4M3 MLP weights, then run
         Masked Calibration Simulation (MCS) to recalibrate all F32
         `input_scale` scalars to observed `amax / 2688.0 * headroom`.

The splice and scale-update helpers are pure-tensor and CPU-testable;
actual NVFP4 forward execution requires the GPU env (modelopt / a
nvfp4-gemm W4A16 kernel) and is guarded accordingly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfm3_dlm.tokenizer import MASK_TOKEN_ID

# NVFP4 amax denominator: FP4 E2M1 max (6.0) x FP8 E4M3 scale max (448.0).
NVFP4_AMAX_DENOMINATOR = 2688.0

# Quantized-tensor sidecar keys shipped by an NVFP4 checkpoint; these are
# owned by the base checkpoint and are never overwritten by a splice.
_SCALE_SUFFIXES = (".input_scale", ".weight_scale", ".weight_scale_2")

FP8_RECIPE = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "scheme": "FP8_DYNAMIC",  # per-channel W8 + dynamic per-row A8; NOT FP8_BLOCK (128x128)
    "targets": ["re:.*feed_forward\\.(w1|w2|w3)$", "re:.*self_attn\\.(q|k|v|o)_proj$"],
    "ignore": [
        "re:.*conv.*",  # conv mixer projections + depthwise conv
        "re:.*gate_proj.*",
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
    ],
    "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel"},
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
    "output_activations": None,
}

MCS_R_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
MCS_NUM_FORWARDS = 256

NVFP4_LOADER_NOTE = """\
# NVFP4Config — W4A16 GEMV loader note
#
# Dispatch contract for `LFM3-1.7B-Diffusion-NVFP4` MLP weights
# (U8-packed FP4 E2M1 pairs + F8_E4M3 block scales + F32 input_scale).
#
# - Kernel: nvfp4-gemm W4A16 GEMV/GEMM; BF16 activations in, BF16 out.
# - Scope: decode shapes only — M <= 2048 rows per GEMM. Above that,
#   fall back to the BF16 path (prefill is already BF16-friendly).
# - Dequant: W = fp4_unpack(U8) * weight_scale(F8_E4M3) * input_scale(F32)
#   applied on the fly inside the kernel; no materialized BF16 weights.
# - KV cache: standard BF16 KV. FP4/FP8 KV is deferred to long-context work.
# - W4A4 activation dispatch is deferred; activations stay BF16 (A16).
# - Attention/conv/norm/embed/lm_head tensors are plain BF16 loads.
# - Fallback: if nvfp4-gemm is unavailable, serve the PerRow-FP8 export.
# - Verify: bit-identical weight checksums vs the NVFP4 base checkpoint;
#   only the 72 F32 input_scale scalars differ after MCS recalibration.
"""


def is_quantized_mlp_key(key: str) -> bool:
    """True for feed_forward weight/scale keys owned by the NVFP4 base."""
    return ".feed_forward." in key


def splice_nvfp4_state_dict(
    base_sd: dict[str, torch.Tensor], trained_sd: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Overlay trained BF16 mixer/norm/embedding tensors onto the NVFP4 base.

    Every `feed_forward.*` key (U8 weights, F8_E4M3 scales, F32 input_scale)
    comes from `base_sd` bit-identically; all other keys present in the
    trained checkpoint are replaced by their BF16 trained values.
    """
    merged = dict(base_sd)
    for k, v in trained_sd.items():
        if is_quantized_mlp_key(k):
            continue  # frozen MLPs stay exactly as the NVFP4 base shipped
        if k not in merged:
            continue  # checkpoint-only aux keys (e.g. fused w13 leftovers)
        merged[k] = v
    return merged


def update_input_scales(
    sd: dict[str, torch.Tensor],
    amax: dict[str, float],
    *,
    headroom: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rewrite every `*.input_scale` scalar to `amax / 2688.0 * headroom`.

    `amax` is keyed by the quantized linear's module prefix (e.g.
    `model.layers.3.feed_forward.w1`). `headroom >= 1.0` guarantees no
    clipping; a larger denominator side is impossible since 2688 already
    encodes the FP4 x FP8 representable range.
    """
    if headroom < 1.0:
        raise ValueError(f"headroom {headroom} < 1.0 would clip calibrated activations")
    out = dict(sd)
    missing = []
    for prefix, a in amax.items():
        key = prefix + ".input_scale"
        if key not in out:
            missing.append(key)
            continue
        out[key] = torch.tensor(a / NVFP4_AMAX_DENOMINATOR * headroom, dtype=torch.float32)
    if missing:
        raise KeyError(f"no input_scale tensor for calibrated modules: {missing}")
    return out


class AmaxCollector:
    """Forward-pre hooks that record the running input amax per module."""

    def __init__(self, module_prefixes: list[str]):
        self.amax = {p: 0.0 for p in module_prefixes}
        self._hooks = []

    def attach(self, named_modules) -> None:
        for name, mod in named_modules:
            if name in self.amax:
                self._hooks.append(mod.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        def fn(_mod, inputs):
            a = inputs[0].detach().abs().max().item()
            if a > self.amax[name]:
                self.amax[name] = a

        return fn

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def mask_inputs(ids: torch.Tensor, r: float, *, generator: torch.Generator) -> torch.Tensor:
    """Replace a fraction `r` of non-special positions with the mask token."""
    out = ids.clone()
    keep = torch.rand(ids.shape, generator=generator) >= r
    out[~keep] = MASK_TOKEN_ID
    return out


def run_mcs(
    model,
    calib_ids: list[torch.Tensor],
    mlp_prefixes: list[str],
    *,
    num_forwards: int = MCS_NUM_FORWARDS,
    r_values=MCS_R_VALUES,
    seed: int = 0,
) -> dict[str, float]:
    """Masked Calibration Simulation driver skeleton.

    Runs `num_forwards` forwards, cycling mask ratios over `r_values`, while
    hooks capture the input amax of every quantized MLP linear. On GPU the
    model must expose NVFP4 weights via modelopt/nvfp4-gemm; this function
    only assumes a `model(input_ids=...)` forward.
    """
    collector = AmaxCollector(list(mlp_prefixes))
    collector.attach(model.named_modules())
    gen = torch.Generator().manual_seed(seed)
    try:
        with torch.no_grad():
            for i in range(num_forwards):
                ids = calib_ids[i % len(calib_ids)]
                r = r_values[i % len(r_values)]
                model(input_ids=mask_inputs(ids, r, generator=gen))
    finally:
        collector.remove()
    return collector.amax


def _load_sd(path: Path) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    if path.is_dir():
        for f in sorted(path.glob("*.safetensors")):
            sd.update(load_safetensors(f))
    elif path.suffix == ".safetensors":
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd


def _save_sd(sd: dict[str, torch.Tensor], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    save_safetensors(sd, dst / "model.safetensors")


def export_bf16(args) -> None:
    from lfm3_dlm.convert import convert_checkpoint

    convert_checkpoint(args.source, args.dst, attention_mode="causal")
    print(f"BF16 export written to {args.dst}")


def export_fp8(args) -> None:
    try:
        import llmcompressor  # noqa: F401
        import compressed_tensors  # noqa: F401
    except ImportError:
        dst = Path(args.dst)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "fp8_recipe.json").write_text(json.dumps(FP8_RECIPE, indent=2))
        print(
            "llmcompressor/compressed-tensors not installed; wrote "
            f"{dst / 'fp8_recipe.json'} for the GPU runbook step to apply "
            "(compressed-tensors FP8_DYNAMIC, per-channel W8 + dynamic per-row A8)."
        )
        return
    raise RuntimeError(
        "FP8 quantization requires a GPU + calibration data; run the GPU "
        "runbook step which loads fp8_recipe.json via llmcompressor.oneshot."
    )


def export_nvfp4(args) -> None:
    base_sd = _load_sd(Path(args.nvfp4_base))
    trained_sd = _load_sd(Path(args.trained))
    merged = splice_nvfp4_state_dict(base_sd, trained_sd)

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "NVFP4_LOADER.md").write_text(NVFP4_LOADER_NOTE)

    if args.amax_json is not None:
        amax = json.loads(Path(args.amax_json).read_text())
        merged = update_input_scales(merged, amax, headroom=args.headroom)
        _save_sd(merged, dst)
        print(f"MCS-NVFP4 export written to {dst} (input_scales recalibrated)")
        return

    try:
        import modelopt  # noqa: F401
    except ImportError:
        _save_sd(merged, dst)
        print(
            "modelopt not installed; wrote spliced weights with UNCALIBRATED "
            f"input_scales to {dst}. GPU step: run run_mcs() over "
            f"{MCS_NUM_FORWARDS} masked forwards (r in {MCS_R_VALUES}), dump "
            "amax JSON, then re-run with --amax-json."
        )
        return
    raise RuntimeError(
        "GPU MCS path: load the spliced checkpoint under modelopt, call "
        "run_mcs() for amax, then update_input_scales() + save."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bf16", help="standalone BF16 export")
    b.add_argument("source")
    b.add_argument("dst")
    b.set_defaults(fn=export_bf16)

    f = sub.add_parser("fp8", help="FP8_DYNAMIC per-channel-W8/dynamic-A8 export")
    f.add_argument("source")
    f.add_argument("dst")
    f.set_defaults(fn=export_fp8)

    n = sub.add_parser("nvfp4", help="MCS-NVFP4 splice export")
    n.add_argument("trained", help="trained BF16 checkpoint (mixer/norm/embed source)")
    n.add_argument("nvfp4_base", help="LiquidAI/LFM3-1.7B-NVFP4 weights dir/file")
    n.add_argument("dst")
    n.add_argument("--amax-json", help="observed per-module amax from a GPU run_mcs pass")
    n.add_argument("--headroom", type=float, default=1.0)
    n.set_defaults(fn=export_nvfp4)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
