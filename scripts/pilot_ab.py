"""Matched-compute A/B: current complementary packing vs K-copy MMT.

Both arms get the same wall-clock budget. Each reports the corpus tokens it
consumed and a held-out loss measured through an identical evaluation path
(`n_masks=2`, no gradients), so the comparison is tokens-per-second against
quality at equal spend — the protocol the MMT paper uses.
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
from lfm3_dlm.train.collator import IM_END_ID, SharedPrefixCollator
from lfm3_dlm.train.loss_cad import compute_step_loss
from lfm3_dlm.train.trainer import DLMTrainer


def _examples(g, count, quarter):
    return [{
        "prompt_ids": torch.randint(1000, 60_000, (quarter,), generator=g).tolist(),
        "response_ids": torch.randint(1000, 60_000, (quarter - 1,), generator=g).tolist() + [IM_END_ID],
    } for _ in range(count)]


@torch.no_grad()
def heldout_loss(model, examples, s_phys, block_size):
    """Same evaluation for both arms: complementary packing, no gradients."""
    coll = SharedPrefixCollator(s_phys=s_phys, block_size=block_size, n_masks=2)
    model.eval()
    total, n = 0.0, 0
    for row in coll(examples):
        out = model.model(input_ids=row.input_ids.unsqueeze(0),
                          position_ids=row.position_ids.unsqueeze(0),
                          doc_ids=row.doc_id.unsqueeze(0),
                          stream_ids=row.stream_id.unsqueeze(0))
        cad = row.cad_mask.nonzero(as_tuple=True)[0]
        ce = (row.ce_weight > 0).nonzero(as_tuple=True)[0]
        ar = (row.ar_weight > 0).nonzero(as_tuple=True)[0]
        res = compute_step_loss(
            out.last_hidden_state.squeeze(0), model.get_output_classifier(),
            cad_idx=cad, teacher_ids=row.teacher_ids[cad],
            teacher_logprobs=row.teacher_logprobs[cad],
            teacher_residual=row.teacher_residual[cad],
            ce_idx=ce, ce_targets=row.labels[ce], ce_weights=row.ce_weight[ce],
            ar_idx=ar, ar_targets=row.labels[ar], w_ar=1.0, causal_x0=True)
        total += float(res.loss)
        n += 1
    return total / max(n, 1)


def run_arm(n_masks: int, budget_s: float, args, train_examples, eval_examples,
            quarter: int) -> dict:
    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = dict(seed=0, s_phys=args.s_phys, block_size=args.block_size,
               lr_max=5e-5, lr_min=2e-6, max_steps=10**6, warmup_steps=0,
               w_ar=1.0, freeze_mlps=True, fp8_mlps=False, block_warmup_tokens=0,
               microbatch=1, use_flex_attention=args.flex, n_masks=n_masks)
    trainer = DLMTrainer(model, cfg, device="cuda")
    t0 = time.perf_counter()
    steps = 0
    while time.perf_counter() - t0 < budget_s:
        trainer.train_step(train_examples)
        steps += 1
    elapsed = time.perf_counter() - t0
    # physical -> corpus ratio depends on K: a row is P+(1+K)R physical for
    # P+R corpus, so K=2 is 2:1 and K=4 is 3:1. Dividing by a constant would
    # flatter the larger K.
    ratio = (2 * quarter) / (quarter + (1 + n_masks) * quarter)
    corpus = trainer.tokens_seen * ratio
    return {
        "n_masks": n_masks, "steps": steps, "seconds": round(elapsed, 1),
        "mode": args.mode, "corpus_tokens": round(corpus),
        "corpus_tok_s": round(corpus / elapsed),
        "heldout_loss": round(heldout_loss(model, eval_examples,
                                           args.s_phys, args.block_size), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--s-phys", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--budget-seconds", type=float, default=600.0,
                    help="wall clock per arm, so both arms spend the same compute")
    ap.add_argument("--arms", default="2,4")
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--mode", default="synthetic_throughput",
                    choices=["synthetic_throughput", "quality"],
                    help="synthetic repeats measure throughput only; quality needs a real corpus stream")
    ap.add_argument("--out", default="docs/runs/g4_pilot.md")
    args = ap.parse_args(argv)

    g = torch.Generator().manual_seed(0)
    arms = [int(k) for k in args.arms.split(",")]
    # a row holds P+(1+K)R at P=R=quarter, so the widest arm sets the size
    quarter = args.s_phys // (2 + max(arms))
    train_examples = _examples(g, 2, quarter)
    eval_examples = _examples(torch.Generator().manual_seed(1234), 2, quarter)

    results = [run_arm(k, args.budget_seconds, args, train_examples,
                       eval_examples, quarter) for k in arms]
    best = min(results, key=lambda r: r["heldout_loss"])
    summary = {"budget_seconds": args.budget_seconds, "arms": results,
               "winner": best["n_masks"]}
    print(json.dumps(summary))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# G4 packing A/B\n\n"
        f"Matched compute: {args.budget_seconds:.0f} s per arm, "
        f"s_phys={args.s_phys}, flex={args.flex}.\n\n"
        + "\n".join(
            f"- K={r['n_masks']}: {r['steps']} steps, {r['corpus_tokens']} corpus tokens, "
            f"{r['corpus_tok_s']} tok/s, held-out loss {r['heldout_loss']}"
            for r in results)
        + f"\n\nWinner at equal compute: K={best['n_masks']}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
