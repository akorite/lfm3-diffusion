"""Phase 1: G4 conversion run (300M corpus tokens over the teacher cache).

Reads the Phase-0 Top-16 cache and a raw corpus stream, packs the
`[P | R_clean | R_m1 | R_m2]` rows through `SharedPrefixCollator`, and
trains with the reused-LSE CAD + CE + AR objective. Every metric lands in
`<out>/metrics.jsonl`; checkpoints follow the config's token interval and
the final Top-5 average is written to `<out>/final/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from lfm3_dlm.data.top16_cache import unpack_top16
from lfm3_dlm.tokenizer import load_tokenizer
from lfm3_dlm.train.trainer import DLMTrainer, cosine_lr


class TeacherCache:
    """Random-access reader over the Phase-0 shards + index."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.entries = [json.loads(line) for line in (self.root / "index.jsonl").open()]
        self._shards: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _shard(self, idx: int) -> np.ndarray:
        if idx not in self._shards:
            raw = np.fromfile(self.root / f"shard-{idx:05d}.bin", dtype=np.uint8)
            self._shards[idx] = raw.reshape(-1, 68)
        return self._shards[idx]

    def example(self, i: int) -> dict:
        e = self.entries[i]
        rows = self._shard(e["shard"])[e["offset"]: e["offset"] + e["R"]]
        ids, lps, res = unpack_top16(torch.from_numpy(rows.copy()))
        return {
            "prompt_ids": e["prompt_ids"],
            "response_ids": e["response_ids"],
            "teacher_ids": ids.t().contiguous(),      # [16, R]
            "teacher_logprobs": lps.t().contiguous(),  # [16, R]
            "teacher_residual": res,                   # [R]
        }


def corpus_examples(dataset_id, split, prompt_field, response_field, tokenizer,
                    seq_len, take):
    """Teacherless examples: the collator supervises them with plain CE."""
    from datasets import load_dataset

    out = []
    for example in load_dataset(dataset_id, split=split, streaming=True):
        prompt = example[prompt_field]
        prompt_msgs = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": prompt}]
        p = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True,
                                          tokenize=True)
        r = tokenizer.encode(str(example[response_field]), add_special_tokens=False)
        r = r + [tokenizer.eos_token_id or 124_900]
        if len(p) + len(r) <= seq_len:
            out.append({"prompt_ids": p, "response_ids": r})
        if len(out) >= take:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/g4_300m.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default=None, help="raw corpus for teacherless rows")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--response-field", default="response")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args(argv)

    from lfm3_dlm import Lfm3ForCausalLM

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = (out / "metrics.jsonl").open("a")

    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    trainer = DLMTrainer(model, cfg, device=device)
    cache = TeacherCache(args.cache)
    corpus = (corpus_examples(args.dataset, args.split, args.prompt_field,
                              args.response_field, tokenizer,
                              cfg.get("s_phys", 4096), 64)
              if args.dataset else [])

    rng = torch.Generator().manual_seed(cfg.get("seed", 0))
    cursor = 0
    t0 = time.perf_counter()
    last_ckpt = trainer.tokens_seen
    while True:
        if args.max_steps and trainer.step >= args.max_steps:
            break
        if trainer.tokens_seen >= cfg.get("total_tokens", 300_000_000):
            break
        if cursor >= len(cache):
            cursor = 0
        batch = []
        for _ in range(args.examples_per_step):
            # 4:1 cache:corpus keeps the CAD term fed while the teacherless
            # rows exercise the CE fallback the OPUT phase relies on
            if corpus and int(torch.randint(0, 5, (1,), generator=rng)) == 0:
                batch.append(corpus[int(torch.randint(0, len(corpus), (1,), generator=rng))])
            else:
                # wrap per draw: a short cache must still fill the batch
                batch.append(cache.example(cursor % len(cache)))
                cursor += 1
        loss = trainer.train_step(batch)
        lr = cosine_lr(trainer.step, cfg["max_steps"], cfg.get("warmup_steps", 0),
                       cfg["lr_max"], cfg["lr_min"])
        row = {
            "step": trainer.step, "tokens_seen": trainer.tokens_seen,
            "loss": loss, "lr": lr,
            "tok_s": trainer.tokens_seen / max(time.perf_counter() - t0, 1e-9),
            "block_size": trainer.collator.block_size,
            "oput": trainer._oput_due(),
        }
        metrics.write(json.dumps(row) + "\n")
        if trainer.step % 50 == 0:
            metrics.flush()
            print(json.dumps(row), flush=True)
        if trainer.tokens_seen - last_ckpt >= cfg.get("checkpoint_interval_tokens", 75_000_000):
            trainer.save_checkpoint(out / f"ckpt-{trainer.tokens_seen}")
            last_ckpt = trainer.tokens_seen

    trainer.save_checkpoint(out / "final-ckpt")
    print(f"done: {trainer.tokens_seen} tokens, {trainer.step} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
