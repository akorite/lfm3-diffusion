"""Measure real token parity (no forced tokens) between the sampler and AR.

For each prompt we take the AR greedy continuation as the reference, then run
the sampler freely and count positions that match. That is the number the
product cares about: a diffusion decode that reproduces AR output 95%+ of the
time is behaviourally lossless without any training.
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


@torch.no_grad()
def ar_reference(model, prompt, n):
    cur = prompt
    out = []
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
    return n / max(len(reference), 1), n


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
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--small-block", type=int, default=4)
    ap.add_argument("--weights", default=None,
                    help="trained state dict to score; omit to score the base model")
    ap.add_argument("--out", default="docs/runs/parity_sweep.md")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.weights:
        load_trained(model, args.weights)
    model = model.cuda().eval()
    model.config.attention_mode = "causal"

    texts = [
        "Explain in two sentences why the sky is blue.",
        "Write a Python function that reverses a linked list.",
        "What is the capital of France, and what river runs through it?",
        "Summarise the plot of Romeo and Juliet in three sentences.",
        "Give one reason a database index speeds up reads.",
        "Translate 'good morning' into Spanish and German.",
    ][: args.prompts]
    prompts = [tok(format_chat(t), return_tensors="pt",
                     add_special_tokens=False).input_ids.cuda() for t in texts]

    configs = [
        ("ar_trust_2", dict(mode="ar_trust", isd_width=2)),
        ("ar_trust_4", dict(mode="ar_trust", isd_width=4)),
        ("ar_trust_8", dict(mode="ar_trust", isd_width=8)),
        ("spd_0.90", dict(mode="spd", hard_commit=0.90)),
        ("spd_0.99", dict(mode="spd", hard_commit=0.99)),
        ("hybrid_auto", dict(mode="hybrid_auto")),
    ]
    rows = []
    for prompt, text in zip(prompts, texts):
        reference = ar_reference(model, prompt, args.tokens)
        for name, cfg in configs:
            res = generate(model, prompt, max_new_tokens=args.tokens,
                           block_size=args.block_size, small_block=args.small_block,
                           **cfg)
            emitted = res.ids[0, prompt.shape[1]:].tolist()[: args.tokens]
            parity, matched = prefix_parity(emitted, reference)
            rows.append({"config": name, "prompt": text[:40], "parity": round(parity, 3),
                         "matched": matched, "tpf": round(float(res.tpf), 2)})
            print(json.dumps(rows[-1]), flush=True)

    summary = {}
    for name, _ in configs:
        sub = [r for r in rows if r["config"] == name]
        summary[name] = {
            "mean_parity": round(sum(r["parity"] for r in sub) / len(sub), 3),
            "min_parity": round(min(r["parity"] for r in sub), 3),
            "mean_tpf": round(sum(r["tpf"] for r in sub) / len(sub), 2),
        }
    print("SUMMARY", json.dumps(summary))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Real token parity (no forced tokens)\n\n"
                   f"{args.prompts} prompts x {args.tokens} tokens, block "
                   f"{args.block_size}/{args.small_block}.\n\n"
                   "| config | mean parity | worst parity | mean TPF |\n|---|---|---|---|\n"
                   + "\n".join(f"| {k} | {v['mean_parity']:.3f} | {v['min_parity']:.3f} | {v['mean_tpf']} |"
                               for k, v in summary.items())
                   + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
