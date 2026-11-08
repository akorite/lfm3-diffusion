"""Compare SPD settings on output quality, not just tokens per forward.

Throughput is easy to buy with a looser commit rule; the question is what the
text looks like when you do. This prints one 64-token sample per setting.

Usage: python scripts/spd_quality.py --weights <w> --out /content/spd_quality.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.sample import generate
from lfm3_dlm.tokenizer import format_chat, load_tokenizer

CONFIGS = [
    (8, 4, 0.9),
    (16, 16, 0.5),
    (32, 16, 0.7),
    (32, 32, 0.3),
]
PROMPT = "Explain why the sky is blue."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default="/content/spd_quality.json")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    model = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B",
                                            dtype=torch.bfloat16)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    model.config.attention_mode = "causal"

    ids = tok(format_chat([{"role": "user", "content": PROMPT}]),
              return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    rows = []
    for block, small, hc in CONFIGS:
        r = generate(model, ids, mode="spd", max_new_tokens=args.tokens,
                     block_size=block, small_block=small, hard_commit=hc)
        em = r.ids[0, ids.shape[1]:].tolist()
        distinct = len(set(em))
        text = tok.decode(em, skip_special_tokens=True)
        rows.append({"block": block, "small": small, "hard_commit": hc,
                     "tpf": round(float(r.tpf), 2), "distinct_tokens": distinct,
                     "of": len(em), "text": text[:400]})
        print("Q", json.dumps(rows[-1])[:600], flush=True)

    ar = generate(model, ids, mode="ar_trust", max_new_tokens=args.tokens,
                  block_size=8, small_block=4, isd_width=4)
    rows.append({"mode": "ar_trust", "tpf": round(float(ar.tpf), 2),
                 "text": tok.decode(ar.ids[0, ids.shape[1]:].tolist(),
                                    skip_special_tokens=True)[:400]})
    print("Q", json.dumps(rows[-1])[:600], flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
