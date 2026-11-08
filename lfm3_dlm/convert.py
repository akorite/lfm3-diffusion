"""Deterministic conversion between a `LiquidAI/LFM3-1.7B`-layout checkpoint
and a standalone, self-contained HF export directory.

The exported directory keeps the *unfused* `feed_forward.w1/w2/w3` layout
(the `w13` fusion is a training-time runtime wrapper only — see
`train/fp8_mlp.py`), carries the mask token bound to row 125033 of the tied
embedding (initialized from the pad row), sets `attention_mode`, and ships
every source file `AutoModelForCausalLM(..., trust_remote_code=True)` needs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from .configuration_lfm3 import Lfm3Config
from .modeling_lfm3 import Lfm3ForCausalLM
from .tokenizer import MASK_TOKEN_ID, load_tokenizer

_PACKAGE_DIR = Path(__file__).parent
# Code files copied so the export loads standalone via trust_remote_code.
_CODE_FILES = (
    "configuration_lfm3.py",
    "modeling_lfm3.py",
    "lfm3_conv.py",
    "conv_halo.py",
    "masks.py",
)
_LICENSE_FILES = ("LICENSE", "NOTICE")

_LICENSE_STUB = (
    "LFM Open License v1.0\n"
    "\n"
    "This checkpoint is a derivative of LiquidAI/LFM3-1.7B and is distributed\n"
    "under the LFM Open License v1.0. See https://www.liquid.ai/lfm-license\n"
    "for the license text. Replace this file with the full license before\n"
    "public distribution.\n"
)


def copy_pad_embedding_to_mask(model: Lfm3ForCausalLM) -> None:
    """Copy the pad-token embedding row onto the `<|mask|>` row (in place).

    Row 125033 is unused upstream, so its random init is meaningless; the pad
    row is a neutral, trained starting point for the mask embedding.
    """
    emb = model.model.embed_tokens.weight
    pad_id = model.config.pad_token_id
    if pad_id is None or pad_id >= emb.shape[0] or MASK_TOKEN_ID >= emb.shape[0]:
        raise ValueError(
            f"need pad_token_id and row {MASK_TOKEN_ID} inside a {emb.shape[0]}-row embedding"
        )
    with torch.no_grad():
        emb[MASK_TOKEN_ID].copy_(emb[pad_id])


def _split_fused_w13(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Undo the training-time `w13` fusion so exports stay unfused."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith("w13.weight"):
            inter = v.shape[0] // 2
            base = k[: -len("w13.weight")]
            out[base + "w1.weight"] = v[:inter]
            out[base + "w3.weight"] = v[inter:]
        elif k.endswith("w13_cm.weight"):
            continue  # transposed copy of w13; redundant for export
        else:
            out[k] = v
    return out


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    if path.is_dir():
        for f in sorted(path.glob("*.safetensors")):
            sd.update(load_safetensors(f))
        for f in sorted(path.glob("*.bin")):
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
        if not sd:
            raise FileNotFoundError(f"no .safetensors/.bin weights under {path}")
    elif path.suffix == ".safetensors":
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    # Tied lm_head rows are dropped on save; re-tie before a strict load.
    if "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd


def _write_export(model: Lfm3ForCausalLM, license_source: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.config.auto_map = {
        "AutoConfig": "configuration_lfm3.Lfm3Config",
        "AutoModel": "modeling_lfm3.Lfm3Model",
        "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM",
    }
    model.save_pretrained(dst)
    for name in _CODE_FILES:
        shutil.copyfile(_PACKAGE_DIR / name, dst / name)

    license_dir = license_source if license_source.is_dir() else license_source.parent
    copied_license = False
    for name in _LICENSE_FILES:
        src_file = license_dir / name
        if src_file.is_file():
            shutil.copyfile(src_file, dst / name)
            copied_license = copied_license or name == "LICENSE"
    if not copied_license:
        (dst / "LICENSE").write_text(_LICENSE_STUB)

    load_tokenizer().save_pretrained(dst)


def convert_checkpoint(
    source: str | Path,
    dst: str | Path,
    *,
    attention_mode: str = "block_diffusion",
    dtype: torch.dtype | None = None,
) -> Path:
    """Convert a source checkpoint into a standalone diffusion-ready export.

    `source` is a directory (or weights file) holding an Lfm3ForCausalLM in the
    unfused `w1/w2/w3` layout. The export is deterministic and idempotent.
    """
    source = Path(source)
    dst = Path(dst)
    config = Lfm3Config.from_pretrained(source if source.is_dir() else source.parent)
    model = Lfm3ForCausalLM(config)
    model.load_state_dict(_split_fused_w13(_load_state_dict(source)), strict=True)
    if dtype is not None:
        model = model.to(dtype)
    model.config.attention_mode = attention_mode
    copy_pad_embedding_to_mask(model)
    _write_export(model, source, dst)
    return dst


def merge_mixer_state(
    base: str | Path,
    trained: str | Path,
    dst: str | Path,
    *,
    attention_mode: str = "block_diffusion",
) -> Path:
    """Merge a trained checkpoint's mixer + norm weights back onto the *base*
    checkpoint's position-wise weights (the inverse direction of conversion).

    `trained` may be a full checkpoint or a mixer-only state dict, with or
    without the `w13` fusion. Frozen `feed_forward` and embedding weights are
    always taken from `base`, so the result is a clean unfused LFM3 layout.
    """
    base = Path(base)
    trained = Path(trained)
    dst = Path(dst)

    model = Lfm3ForCausalLM.from_pretrained(base)
    base_sd = model.state_dict()
    trained_sd = _split_fused_w13(_load_state_dict(trained))

    merged = dict(base_sd)
    for k, v in trained_sd.items():
        if k not in base_sd:
            continue  # checkpoint-only aux keys (e.g. w13 leftovers)
        if ".feed_forward." in k:
            continue  # frozen MLPs stay exactly as the base shipped them
        merged[k] = v
    model.load_state_dict(merged, strict=True)
    model.config.attention_mode = attention_mode
    copy_pad_embedding_to_mask(model)
    _write_export(model, base, dst)
    return dst


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LFM3 block-diffusion checkpoint converter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="convert a source checkpoint into a standalone export")
    e.add_argument("source")
    e.add_argument("dst")
    e.add_argument("--attention-mode", default="block_diffusion")
    m = sub.add_parser("merge", help="merge trained mixer weights back onto a base checkpoint")
    m.add_argument("base")
    m.add_argument("trained")
    m.add_argument("dst")
    m.add_argument("--attention-mode", default="block_diffusion")
    args = ap.parse_args()
    if args.cmd == "export":
        convert_checkpoint(args.source, args.dst, attention_mode=args.attention_mode)
    else:
        merge_mixer_state(args.base, args.trained, args.dst, attention_mode=args.attention_mode)


if __name__ == "__main__":
    main()


__all__ = ["convert_checkpoint", "copy_pad_embedding_to_mask", "merge_mixer_state"]
