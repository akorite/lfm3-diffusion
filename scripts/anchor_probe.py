"""How much anchor does the denoiser need before its fills agree with reality?

Takes a real continuation, reveals the first r tokens of a block, masks the
rest, and measures how often the argmax at the masked slots matches the true
continuation. Steep improvement with r means the denoiser works when it has
context and the parallel collapse is the documented dependency problem; a flat
curve means the fill quality itself is the ceiling.

Usage: python scripts/anchor_probe.py --weights <w> --out /content/anchor.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.tokenizer import MASK_TOKEN_ID, format_chat, load_tokenizer

OUTPUT_VOCAB = 125_033
PROMPTS = [
    "Explain why the sky is blue.",
    "Give one reason a database index speeds up reads.",
    "Write a Python function that reverses a linked list.",
]


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default="/content/anchor.json")
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
    w = model.get_output_classifier()

    rows = []
    for text in PROMPTS:
        ids = tok(format_chat([{"role": "user", "content": text}]),
                  return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        cur, cont = ids, []
        with torch.no_grad():
            for _ in range(args.tokens):
                t = int(model(input_ids=cur).logits[0, -1, :OUTPUT_VOCAB].argmax())
                cont.append(t)
                cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], 1)
        P = ids.shape[1]
        for reveal in (0, 2, 4, 8, 16, args.tokens):
            body = [MASK_TOKEN_ID] * args.tokens
            for j in range(reveal):
                body[j] = cont[j]
            x = torch.cat([ids, torch.tensor([body], device=ids.device)], 1)
            with torch.no_grad():
                hidden = model.model(input_ids=x).last_hidden_state[0]
            logits = hidden[P - 1: P - 1 + args.tokens] @ w.t()
            pred = logits[:, :OUTPUT_VOCAB].argmax(-1).tolist()
            start = reveal
            agree = sum(int(pred[j] == cont[j]) for j in range(start, args.tokens))
            rows.append({"prompt": text[:28], "reveal": reveal,
                         "agree": round(agree / max(1, args.tokens - start), 3),
                         "distinct": len(set(pred[start:]))})
    Path(args.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print("A", r["prompt"][:18], "reveal", r["reveal"],
              "agree", r["agree"], "distinct", r["distinct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
