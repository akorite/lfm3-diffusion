"""Causal-mode perplexity on held-out corpus rows, trained vs base.

The diffusion held-out loss says whether the conversion learned its objective;
it says nothing about whether the model still answers like the base LFM3.
This scores the same kind of rows with plain next-token cross-entropy in
`causal` mode, which is the number a reader would compare against the
published model.

Usage: python scripts/causal_ppl.py --weights <w> --count 40 --out <json>
       python scripts/causal_ppl.py --count 40            # base model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from heldout_eval import build_examples
from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.tokenizer import load_tokenizer

OUTPUT_VOCAB = 125_033


@torch.no_grad()
def mean_nll(model, examples, max_len):
    model.eval()
    total, tokens = 0.0, 0
    for ex in examples:
        ids = ex["prompt_ids"] + ex["response_ids"]
        if len(ids) > max_len:
            continue
        x = torch.tensor([ids], device="cuda")
        logits = model(input_ids=x).logits[0, :-1, :OUTPUT_VOCAB]
        target = x[0, 1:]
        start = len(ex["prompt_ids"]) - 1
        ce = torch.nn.functional.cross_entropy(
            logits[start:], target[start:], reduction="sum")
        total += float(ce)
        tokens += target.shape[0] - start
    return total / max(tokens, 1), tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None,
                    help="trained state dict; omit to score the base model")
    ap.add_argument("--data-file", default="/content/data/chat.jsonl")
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", default="/content/causal_ppl.json")
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

    examples = build_examples(args.data_file, args.start, args.count, tok, 4096)
    nll, tokens = mean_nll(model, examples, args.max_len)
    result = {"weights": args.weights, "examples": len(examples),
              "scored_tokens": tokens, "mean_nll": round(nll, 4),
              "perplexity": round(float(torch.exp(torch.tensor(nll))), 3)}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("PPL", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
