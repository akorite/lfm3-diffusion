"""Collect on-policy draft-head training pairs.

At decode the head sees the model's *own* text, so the training pairs must come
from the model's own greedy rollouts, not from the corpus: hidden state at each
step of a rollout, with the next `k` tokens the model itself went on to choose.
That is the one difference between this and the teacher-forced run that scored
below the mask-draft.

Writes /content/draft_pairs.pt: {"hidden": [N,H] bf16, "targets": [N,k] long}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.tokenizer import format_chat, load_tokenizer

OUTPUT_VOCAB = 125_033


def load_prompts(path, n, tok):
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                import json as _j
                d = _j.loads(line)
            except Exception:
                continue
            text = str(d.get("input", ""))[:400]
            if not text.strip():
                continue
            out.append(tok(format_chat([{"role": "user", "content": text}]),
                           return_tensors="pt",
                           add_special_tokens=False).input_ids[0].tolist())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data-file", default="/content/data/chat.jsonl")
    ap.add_argument("--prompts", type=int, default=120)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="/content/draft_pairs.pt")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    prompts = load_prompts(args.data_file, args.prompts, tok)
    base = args.weights if (args.weights and not args.weights.endswith(".pt")) \
        else "LiquidAI/LFM3-1.7B"
    model = Lfm3ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if args.weights and args.weights.endswith(".pt"):
        state = torch.load(args.weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    model.config.attention_mode = "causal"

    lat, tgt = [], []
    for pi, p in enumerate(prompts):
        ids = torch.tensor([p], device="cuda")
        hiddens, picks = [], []
        cur = ids
        with torch.no_grad():
            for _ in range(args.rollout + args.k):
                out = model.model(input_ids=cur, output_hidden_states=True)
                h = out.last_hidden_state[0, -1]
                tok_id = int(model.lm_head(h)[:OUTPUT_VOCAB].argmax())
                hiddens.append(h.detach())
                picks.append(tok_id)
                cur = torch.cat([cur, torch.tensor([[tok_id]], device="cuda")], 1)
        for i in range(args.rollout):
            if i + args.k > len(picks):
                break
            lat.append(hiddens[i])
            tgt.append(picks[i:i + args.k])
        if pi % 20 == 0:
            print(json.dumps({"prompt": pi, "pairs": len(lat)}), flush=True)

    hidden = torch.stack(lat).to(torch.bfloat16).cpu()
    targets = torch.tensor(tgt, dtype=torch.long)
    torch.save({"hidden": hidden, "targets": targets, "k": args.k}, args.out)
    print(json.dumps({"pairs": len(lat), "hidden": list(hidden.shape),
                      "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
