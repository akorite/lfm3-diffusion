"""Train the multi-token draft head on a frozen LFM3.

The head only proposes; AR-Trust's verifier still decides every token, so this
cannot change what the model says - only how many tokens one forward yields.

Targets are the corpus continuation, which is what the verifier will accept
whenever the model is right; the head's job is to be right at the same places.

Usage:
  python scripts/train_draft.py --weights <w> --steps 800 --out /content/draft_head.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.sample.draft_head import DraftHead
from lfm3_dlm.tokenizer import load_tokenizer

OUTPUT_VOCAB = 125_033


def load_texts(path, n, tok, max_len):
    rows = []
    with open(path) as f:
        for line in f:
            if len(rows) >= n:
                break
            try:
                import json as _j
                d = _j.loads(line)
            except Exception:
                continue
            text = f"{d.get('input','')}\n{d.get('output','')}"
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > max_len:
                rows.append(ids[:max_len])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data-file", default="/content/data/chat.jsonl")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/content/draft_head.pt")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    rows = load_texts(args.data_file, args.rows, tok, args.seq)
    if not rows:
        print(json.dumps({"error": "no rows"}))
        return 1

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
    for p in model.parameters():
        p.requires_grad = False
    lm_head = model.lm_head
    H = model.config.hidden_size

    head = DraftHead(H, k=args.k).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 20)

    g = torch.Generator().manual_seed(0)
    log = open("/content/draft_train.jsonl", "w")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(rows), (args.batch,), generator=g).tolist()
        xs = [rows[i] for i in idx]
        L = max(len(x) for x in xs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        batch = torch.full((args.batch, L), pad, dtype=torch.long, device="cuda")
        for b, x in enumerate(xs):
            batch[b, :len(x)] = torch.tensor(x, device="cuda")
        with torch.no_grad():
            hidden = model.model(input_ids=batch).last_hidden_state
        # sample positions with k following tokens available
        loss_terms = []
        for b, x in enumerate(xs):
            pos = torch.randint(0, len(x) - args.k - 1, (args.positions,))
            h = hidden[b, pos]
            logits = head(h, lm_head)                      # [n, k, V]
            tgt = torch.stack([batch[b, pos + j + 1] for j in range(args.k)], 1)
            loss_terms.append(F.cross_entropy(
                logits[:, :, :OUTPUT_VOCAB].transpose(1, 2), tgt))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss), 4)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    torch.save({"head": head.state_dict(), "k": args.k}, args.out)
    print(json.dumps({"saved": args.out, "k": args.k, "steps": args.steps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
