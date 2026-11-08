"""Measure the training step against the plan's gate.

Reports corpus tokens/s and peak VRAM for a fixed packed-row shape, and
exits non-zero when either gate is missed so it can gate a run in CI.
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
from lfm3_dlm.train.collator import IM_END_ID
from lfm3_dlm.train.trainer import DLMTrainer

GATE_CORPUS_TOK_S = 22_000
GATE_VRAM_GB = 65.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--s-phys", type=int, default=4096)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--checkpointing", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = dict(seed=0, s_phys=args.s_phys, block_size=32, lr_max=5e-5,
               lr_min=2e-6, max_steps=10**6, warmup_steps=0, w_ar=1.0,
               freeze_mlps=True, fp8_mlps=False, block_warmup_tokens=0,
               microbatch=args.microbatch, use_flex_attention=args.flex,
               activation_checkpointing=args.checkpointing)
    trainer = DLMTrainer(model, cfg, device=device)

    # one example fills a quarter of a row: P = R = s_phys/4
    g = torch.Generator().manual_seed(0)
    quarter = args.s_phys // 4
    examples = [{
        "prompt_ids": torch.randint(1000, 60_000, (quarter,), generator=g).tolist(),
        "response_ids": torch.randint(1000, 60_000, (quarter - 1,), generator=g).tolist() + [IM_END_ID],
    } for _ in range(4 * args.microbatch)]

    for _ in range(args.warmup):
        trainer.train_step(examples)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        trainer.train_step(examples)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rows = 4 * args.microbatch  # two examples per packed row
    corpus_tokens = rows * (args.s_phys // 2)
    corpus_tok_s = corpus_tokens * args.steps / elapsed
    vram = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    result = {
        "device": device, "s_phys": args.s_phys, "microbatch": args.microbatch,
        "flex": args.flex, "checkpointing": args.checkpointing,
        "s_per_step": round(elapsed / args.steps, 3),
        "corpus_tok_s": round(corpus_tok_s), "peak_vram_gb": round(vram, 1),
        "gate_corpus_tok_s": GATE_CORPUS_TOK_S, "gate_vram_gb": GATE_VRAM_GB,
    }
    result["passes"] = corpus_tok_s >= GATE_CORPUS_TOK_S and vram < GATE_VRAM_GB
    print(json.dumps(result))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
