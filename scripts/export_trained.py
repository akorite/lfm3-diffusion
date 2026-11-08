"""Export a trained run directory into a standalone HF checkpoint.

`train_g4.py` writes `final-weights.pt` plus a resumable bundle; this turns
that into the layout `lfm3_dlm.convert` expects and then merges the trained
mixer/norm weights onto the untouched base, so the export is a clean unfused
LFM3 directory that loads with `Lfm3ForCausalLM.from_pretrained`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.convert import merge_mixer_state
from lfm3_dlm.tokenizer import load_tokenizer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="train_g4 --out directory")
    ap.add_argument("--base", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--attention-mode", default="causal")
    args = ap.parse_args(argv)

    run = Path(args.run)
    weights = run / "final-weights.pt"
    if not weights.exists():
        raise SystemExit(f"no trained weights at {weights}")

    # stage the trained weights as a loadable checkpoint for the merge step
    staged = run / "hf-trained"
    staged.mkdir(exist_ok=True)
    torch.save(torch.load(weights, map_location="cpu"), staged / "model.bin")
    Lfm3Config.from_pretrained(args.base).save_pretrained(staged)

    out = merge_mixer_state(args.base, staged, args.dst,
                            attention_mode=args.attention_mode)
    load_tokenizer().save_pretrained(out)
    print(f"exported {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
