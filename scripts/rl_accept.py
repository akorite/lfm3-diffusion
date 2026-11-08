"""Acceptance-targeted self-distillation: teach the denoiser the model's own AR pick.

AR-Trust commits a draft only when the verifier's causal argmax agrees with it,
so the quantity that sets throughput is `argmax(denoise | prefix+MASKs)` matching
`argmax(causal | prefix)`. The supervised objective never targets that pair
directly — it distils the teacher over corrupted streams — which is why drafts
are rejected long after parity is exact.

This phase optimises the agreement itself, expert-iteration style: the model's
own greedy continuation is the expert, the masked-block prediction is the
policy, and cross-entropy on the masked slots is the reward signal. The verifier
is untouched, so AR-greedy parity stays exact by construction.

Usage:
  python scripts/rl_accept.py --weights <w> --steps 1500 --out /content/run5/final-weights.pt
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
from lfm3_dlm.tokenizer import MASK_TOKEN_ID, format_chat, load_tokenizer

OUTPUT_VOCAB = 125_033


def load_rows(path, start, count, tok):
    import json as _json

    rows, seen = [], 0
    with open(path) as f:
        for line in f:
            if seen < start:
                seen += 1
                continue
            if len(rows) >= count:
                break
            seen += 1
            try:
                d = _json.loads(line)
            except Exception:
                continue
            msgs = [{"role": "user", "content": str(d["input"])}]
            if tok.chat_template:
                p = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=True)
            else:
                p = tok.encode(format_chat(msgs), add_special_tokens=False)
            r = tok.encode(str(d["output"]), add_special_tokens=False)
            if len(p) + 16 <= 1024 and len(r) >= 8:
                rows.append((p, r))
    return rows


def build_batch(rows, idx, k):
    return [(rows[i][0], rows[i][1][:k]) for i in idx], k


@torch.no_grad()
def greedy_targets(model, examples, device, draft_row=False):
    """The model's own next-token picks for the first k response slots.

    With `draft_row`, the prefix matches the sampler's draft feed: the prompt,
    then the token the draft is conditioned on, then the response tokens the
    targets come from.
    """
    model.config.attention_mode = "causal"
    targets = []
    for p, rr in examples:
        lead = rr[:1] if draft_row else []
        body = rr[1:] if draft_row else rr
        x = torch.tensor([p + lead + body], device=device)
        hidden = model.model(input_ids=x).last_hidden_state[0]
        w = model.get_output_classifier()
        start = len(p) + len(lead) - 1
        logits = hidden[start: start + len(body)] @ w.t()
        targets.append(logits[:, :OUTPUT_VOCAB].argmax(-1).tolist())
    return targets


def mask_loss(model, examples, targets, device, block=4, draft_row=False):
    model.config.attention_mode = "causal" if draft_row else "block_diffusion"
    losses, correct, total = [], 0, 0
    for (p, rr), tgt in zip(examples, targets):
        k = len(tgt)
        lead = rr[:1] if draft_row else []
        ids = p + lead + [MASK_TOKEN_ID] * k
        if draft_row:
            hidden = model.model(
                input_ids=torch.tensor([ids], device=device)).last_hidden_state[0]
            start = len(p) + len(lead) - 1
        else:
            n = len(ids)
            pos = torch.arange(n, device=device).unsqueeze(0)
            doc = torch.zeros_like(pos)
            sid = torch.cat([torch.zeros(1, len(p), dtype=torch.long, device=device),
                             torch.ones(1, k, dtype=torch.long, device=device)], 1)
            hidden = model.model(input_ids=torch.tensor([ids], device=device),
                                 position_ids=pos, doc_ids=doc,
                                 stream_ids=sid).last_hidden_state[0]
            start = len(p) - 1
        w = model.get_output_classifier()
        logits = hidden[start: start + k] @ w.t()
        t = torch.tensor(tgt, device=device)
        losses.append(F.cross_entropy(logits, t))
        correct += int((logits[:, :OUTPUT_VOCAB].argmax(-1) == t).sum())
        total += k
    return torch.stack(losses).mean(), correct / max(total, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--weights", required=True, help="supervised checkpoint to warm start")
    ap.add_argument("--data-file", default="/content/data/chat.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--row-mode", choices=["noisy", "draft"], default="noisy",
                    help="draft mirrors the sampler's causal mask feed")
    ap.add_argument("--out", default="/content/run5/final-weights.pt")
    ap.add_argument("--log", default="/content/rl_accept.jsonl")
    args = ap.parse_args(argv)

    device = "cuda"
    tok = load_tokenizer()
    rows = load_rows(args.data_file, args.start, args.rows, tok)
    if not rows:
        print(json.dumps({"error": "no rows"}))
        return 1

    model = Lfm3ForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    model = model.cuda()
    # the supervised run freezes MLPs; keep that so this phase cannot undo it
    for name, prm in model.named_parameters():
        prm.requires_grad = "mlp" not in name
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 10)

    log = open(args.log, "w")
    g = torch.Generator().manual_seed(0)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(rows), (args.batch,), generator=g).tolist()
        examples, _ = build_batch(rows, idx, args.block)
        draft_row = args.row_mode == "draft"
        targets = greedy_targets(model, examples, device, draft_row)
        loss, agree = mask_loss(model, examples, targets, device, args.block, draft_row)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss), 4), "mask_argmax_agrees": round(agree, 3)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(json.dumps({"saved": args.out, "steps": args.steps, "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
