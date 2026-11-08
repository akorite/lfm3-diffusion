"""Train the draft head on cached on-policy pairs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.sample.draft_head import DraftHead

OUTPUT_VOCAB = 125_033


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="/content/draft_pairs.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="/content/draft_head_online.pt")
    args = ap.parse_args(argv)

    d = torch.load(args.pairs, map_location="cpu")
    hidden, targets, k = d["hidden"], d["targets"], d["k"]
    print(json.dumps({"pairs": len(hidden), "k": k}), flush=True)

    base = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B",
                                           dtype=torch.bfloat16)
    lm_head = base.lm_head.cuda()
    for p in lm_head.parameters():
        p.requires_grad = False
    H = base.config.hidden_size

    head = DraftHead(H, k=k).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 20)

    hid = hidden.cuda()
    tgt = targets.cuda()
    g = torch.Generator().manual_seed(0)
    log = open("/content/draft_online.jsonl", "w")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(hid), (args.batch,), generator=g)
        logits = head(hid[idx], lm_head)
        loss = F.cross_entropy(logits[:, :, :OUTPUT_VOCAB].transpose(1, 2), tgt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt[idx]).float().mean()
                acc1 = (logits[:, 0].argmax(-1) == tgt[idx][:, 0]).float().mean()
            rec = {"step": step, "loss": round(float(loss), 4),
                   "acc_all": round(float(acc), 3), "acc_t1": round(float(acc1), 3)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    torch.save({"head": head.state_dict(), "k": k}, args.out)
    print(json.dumps({"saved": args.out, "k": k}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
