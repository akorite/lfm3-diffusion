"""Sweep SPD decode knobs for tokens-per-forward, the diffusion-mode number.

AR-Trust is capped near AR speed because it only commits what the causal
verifier accepts. SPD is the diffusion decoder: it denoises a whole block in one
forward and commits the confident part, which is how LLaDA-class models get
their throughput. This reports tokens per forward (and the AR-parity it costs)
across block geometry and the commit threshold, plus one sample per setting so
the output can be eyeballed rather than trusted.

Usage: python scripts/spd_speed.py --weights <w> --out /content/spd_speed.json
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

OUTPUT_VOCAB = 125_033
PROMPTS = [
    "Explain why the sky is blue.",
    "Give one reason a database index speeds up reads.",
    "Write a Python function that reverses a linked list.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
]


def load_trained(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    _, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded <= 0:
        raise SystemExit(f"{path}: loaded 0 tensors")
    print(f"loaded {loaded}/{len(state)} tensors from {path}", flush=True)


@torch.no_grad()
def ar_reference(model, ids, n):
    cur, ref = ids, []
    for _ in range(n):
        t = int(model(input_ids=cur).logits[0, -1, :OUTPUT_VOCAB].argmax())
        ref.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], 1)
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--out", default="/content/spd_speed.json")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    model = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B",
                                            dtype=torch.bfloat16)
    if args.weights:
        load_trained(model, args.weights)
    model = model.cuda().eval()
    model.config.attention_mode = "causal"

    texts = PROMPTS[: args.prompts]
    refs, starts = [], []
    for text in texts:
        ids = tok(format_chat([{"role": "user", "content": text}]),
                  return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        starts.append(ids)
        refs.append(ar_reference(model, ids, args.tokens))

    rows = []
    for block in (16, 32):
        for small in (8, 16, 32):
            if small > block:
                continue
            for hc in (0.3, 0.5, 0.7, 0.9):
                par, tpf, sample = [], [], None
                for ids, ref in zip(starts, refs):
                    r = generate(model, ids, mode="spd", max_new_tokens=args.tokens,
                                 block_size=block, small_block=small,
                                 hard_commit=hc)
                    em = r.ids[0, ids.shape[1]:].tolist()[: args.tokens]
                    n = 0
                    for a, b in zip(em, ref):
                        if a != b:
                            break
                        n += 1
                    par.append(n / args.tokens)
                    tpf.append(float(r.tpf))
                    if sample is None:
                        sample = tok.decode(em, skip_special_tokens=True)[:160]
                rows.append({"block": block, "small": small, "hard_commit": hc,
                             "mean_tpf": round(sum(tpf) / len(tpf), 2),
                             "mean_parity": round(sum(par) / len(par), 3),
                             "sample": sample})
                print(json.dumps(rows[-1])[:300], flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    best = max(rows, key=lambda r: r["mean_tpf"])
    print("BEST", json.dumps(best)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
