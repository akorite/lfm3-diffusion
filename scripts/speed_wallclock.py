"""Wall-clock decoding speed, which is what a user actually feels.

Tokens-per-forward is a shape metric; this times the same 64-token generation
in three modes on the delivered weights so the numbers are in seconds.

Usage: python scripts/speed_wallclock.py --weights <w> --out /content/speed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.sample import generate
from lfm3_dlm.tokenizer import format_chat, load_tokenizer

PROMPT = "Explain why the sky is blue."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default="/content/speed.json")
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
    # plain AR greedy, the honest baseline for wall clock
    torch.cuda.synchronize()
    t0 = time.time()
    cur = ids
    with torch.no_grad():
        for _ in range(args.tokens):
            nxt = torch.tensor([[int(model(input_ids=cur).logits[0, -1,
                                                          :125033].argmax())]],
                               device=cur.device)
            cur = torch.cat([cur, nxt], 1)
    torch.cuda.synchronize()
    ar_s = time.time() - t0
    rows.append({"mode": "ar_greedy", "seconds": round(ar_s, 2),
                 "tok_s": round(args.tokens / ar_s, 1), "tpf": 1.0})

    for name, kw in (("ar_trust", dict(mode="ar_trust", isd_width=4)),
                     ("spd_32_32", dict(mode="spd", block_size=32, small_block=32,
                                        hard_commit=0.3)),
                     ("spd_8_4", dict(mode="spd", block_size=8, small_block=4,
                                      hard_commit=0.9))):
        torch.cuda.synchronize()
        t0 = time.time()
        r = generate(model, ids, max_new_tokens=args.tokens, **kw)
        torch.cuda.synchronize()
        s = time.time() - t0
        rows.append({"mode": name, "seconds": round(s, 2),
                     "tok_s": round(args.tokens / s, 1), "tpf": round(float(r.tpf), 2)})
        print("W", json.dumps(rows[-1]), flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print("WALL", json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
