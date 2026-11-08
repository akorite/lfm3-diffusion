"""Load an exported checkpoint and assert the properties the run promises.

The export step can succeed while producing a directory that no longer loads
like an LFM3 model, or that lost the input embedding row the diffusion sampler
needs. This checks both, so `FINAL.json` reporting `rc=0` is not the only
evidence that the artifact is usable.

Usage: python scripts/assert_export.py --path /content/export_dry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM

MASK_ROW = 125033
PAD_ROW = 124893


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    args = ap.parse_args(argv)

    model = Lfm3ForCausalLM.from_pretrained(args.path, dtype=torch.bfloat16)
    emb = model.model.embed_tokens.weight
    report = {
        "path": args.path,
        "n_layers": len(model.model.layers),
        "vocab": int(emb.shape[0]),
        "mask_row_allclose_pad": bool(
            torch.allclose(emb[MASK_ROW], emb[PAD_ROW])
        ),
        "dtype": str(emb.dtype),
    }
    ok = (report["n_layers"] == 24 and not report["mask_row_allclose_pad"]
          and report["vocab"] > MASK_ROW)
    report["ok"] = ok
    print("ASSERT", json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
