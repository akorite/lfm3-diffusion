"""Sweep the SPD decode knobs against AR greedy parity.

The early probe showed SPD at 6% parity and `hard_commit` 0.90 vs 0.99 making no
difference, which points at the block-end bonus and the two-step top-1 rule
firing before the threshold does rather than at the threshold itself. This
sweep varies block size and commit threshold together so the report can name a
config, or say plainly that none reaches the bar.
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
from lfm3_dlm.tokenizer import load_tokenizer

OUTPUT_VOCAB = 125_033


@torch.no_grad()
def ar_reference(model, prompt, n):
    cur, out = prompt, []
    for _ in range(n):
        tok = int(model(input_ids=cur).logits[0, -1, :OUTPUT_VOCAB].argmax())
        out.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]], device=cur.device)], 1)
    return out


def prefix_parity(emitted, reference):
    n = 0
    for a, b in zip(emitted, reference):
        if a != b:
            break
        n += 1
    return n / max(len(reference), 1)


def load_trained(model, path):
    """Load either a plain state dict or a trainer `checkpoint.pt` bundle.

    A silent 0-key load would measure the base model while reporting the
    checkpoint's name, so the load count is checked.
    """
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    if loaded <= 0:
        raise SystemExit(f"{path}: loaded 0 tensors (missing {len(missing)})")
    print(f"loaded {loaded}/{len(state)} tensors from {path}", flush=True)
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--out", default="/content/spd_sweep.json")
    args = ap.parse_args(argv)

    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.weights:
        load_trained(model, args.weights)
    model = model.cuda().eval()
    model.config.attention_mode = "causal"
    tok = load_tokenizer()

    texts = ["Explain why the sky is blue.",
             "Give one reason a database index speeds up reads."]
    prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
    refs = [ar_reference(model, p, args.tokens) for p in prompts]

    rows = []
    for block in (4, 8, 16, 32):
        small = max(1, block // 2)
        for hc in (0.55, 0.75, 0.90, 0.99, 1.01):
            par, tpfs = [], []
            for p, ref in zip(prompts, refs):
                res = generate(model, p, mode="spd", max_new_tokens=args.tokens,
                               block_size=block, small_block=small, hard_commit=hc)
                emitted = res.ids[0, p.shape[1]:].tolist()[: args.tokens]
                par.append(prefix_parity(emitted, ref))
                tpfs.append(float(res.tpf))
            rows.append({"block": block, "hard_commit": hc,
                         "mean_parity": round(sum(par) / len(par), 3),
                         "mean_tpf": round(sum(tpfs) / len(tpfs), 2)})
            print(json.dumps(rows[-1]), flush=True)

    best = max(rows, key=lambda r: (r["mean_parity"], r["mean_tpf"]))
    passing = [r for r in rows if r["mean_parity"] >= 0.95]
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "best": best,
         "clears_95": bool(passing),
         "best_at_95": max(passing, key=lambda r: r["mean_tpf"]) if passing else None},
        indent=1))
    print("SWEEP", json.dumps({"best": best, "clears_95": bool(passing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
