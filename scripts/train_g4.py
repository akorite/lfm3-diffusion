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

    def __init__(self, root: str | Path, s_phys: int = 4096,
                 block_size: int = 32, n_masks: int = 2):
        self.root = Path(root)
        self.s_phys = s_phys
        self.block_size = block_size
        self.n_masks = n_masks
        self.entries = [json.loads(line) for line in (self.root / "index.jsonl").open()]
        self._shards: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _shard(self, idx: int) -> np.ndarray:
        if idx not in self._shards:
            raw = np.fromfile(self.root / f"shard-{idx:05d}.bin", dtype=np.uint8)
            self._shards[idx] = raw.reshape(-1, 68)
        return self._shards[idx]

    def example(self, i: int) -> dict | None:
        e = self.entries[i]
        # a cache built before the packing bound was enforced can hold rows the
        # collator refuses; skip them instead of crashing the run
        r = max(self.block_size, -(-e["R"] // self.block_size) * self.block_size)
        if e["P"] + (1 + self.n_masks) * r > self.s_phys:
            return None
        rows = self._shard(e["shard"])[e["offset"]: e["offset"] + e["R"]]
        ids, lps, res = unpack_top16(torch.from_numpy(rows.copy()))
        return {
            "prompt_ids": e["prompt_ids"],
            "response_ids": e["response_ids"],
            "teacher_ids": ids.t().contiguous(),      # [16, R]
            "teacher_logprobs": lps.t().contiguous(),  # [16, R]
            "teacher_residual": res,                   # [R]
        }


def _packs(p, r, s_phys, block_size, n_masks=2):
    """Same bound the collator enforces: P + (1+K)R_rounded must fit one row."""
    rr = max(block_size, -(-len(r) // block_size) * block_size)
    return len(p) + (1 + n_masks) * rr <= s_phys


def corpus_examples(dataset_id, split, prompt_field, response_field, tokenizer,
                    seq_len, take, config_name=None, data_files=None,
                    block_size=32):
    """Teacherless examples: the collator supervises them with plain CE."""
    from datasets import load_dataset

    out = []
    if data_files:
        ds = load_dataset("json", data_files=data_files, split="train", streaming=True)
    elif config_name:
        ds = load_dataset(dataset_id, config_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    for example in ds:
        prompt = example[prompt_field]
        prompt_msgs = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": prompt}]
        if tokenizer.chat_template:
            p = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True,
                                              tokenize=True)
        else:
            from lfm3_dlm.tokenizer import format_chat
            p = tokenizer.encode(format_chat(prompt_msgs), add_special_tokens=False)
        r = tokenizer.encode(str(example[response_field]), add_special_tokens=False)
        r = r + [tokenizer.eos_token_id or 124_900]
        if _packs(p, r, seq_len, block_size):
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
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--data-files", default=None)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--response-field", default="response")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint under --out")
    args = ap.parse_args(argv)

    from lfm3_dlm import Lfm3ForCausalLM

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = (out / "metrics.jsonl").open("a")

    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lfm3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.max_steps:
        # the LR schedule must anneal to lr_min at the step we actually stop
        cfg["max_steps"] = args.max_steps
    trainer = DLMTrainer(model, cfg, device=device)
    cache = TeacherCache(args.cache, cfg.get("s_phys", 4096),
                         cfg.get("block_size", 32), cfg.get("n_masks", 2))
    corpus = (corpus_examples(args.dataset, args.split, args.prompt_field,
                              args.response_field, tokenizer,
                              cfg.get("s_phys", 4096), 64, args.dataset_config,
                              args.data_files, cfg.get("block_size", 32))
              if args.dataset else [])

    if args.resume:
        def _ckpt_key(path):
            tail = path.name.split("-")[-1]
            return (0, int(tail)) if tail.isdigit() else (-1, 0)
        ckpts = sorted(out.glob("ckpt-*"), key=_ckpt_key) + (
            [out / "final-ckpt"] if (out / "final-ckpt").exists() else [])
        ckpts = [c for c in ckpts if (c / "checkpoint.pt").exists()]
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
            print(f"resumed from {ckpts[-1]} at step {trainer.step}", flush=True)
        else:
            print("nothing to resume; starting fresh", flush=True)

    rng = torch.Generator().manual_seed(cfg.get("seed", 0))
    cursor = 0
    t0 = time.perf_counter()
    last_ckpt = trainer.tokens_seen
    while True:
        if args.max_steps and trainer.step >= args.max_steps:
            break
        # `tokens_seen` counts physical lanes; the step budget is the schedule
        if args.max_steps == 0 and trainer.tokens_seen >= cfg.get("total_tokens", 3 * 10**8):
            break
        if cursor >= len(cache):
            cursor = 0
        batch = []
        for _ in range(args.examples_per_step):
            # cache:corpus mix keeps the CAD term fed while teacherless rows
            # widen the data; the ratio is a knob because more cache epochs on
            # a small slice is what regressed the 50M checkpoint
            ratio = int(cfg.get("cache_corpus_ratio", 4))
            if corpus and int(torch.randint(0, ratio + 1, (1,), generator=rng)) == 0:
                batch.append(corpus[int(torch.randint(0, len(corpus), (1,), generator=rng))])
            else:
                # wrap per draw: a short cache must still fill the batch
                ex = cache.example(cursor % len(cache))
                cursor += 1
                if ex is None:
                    continue
                batch.append(ex)
        if not batch:
            break
        loss = trainer.train_step(batch)
        lr = cosine_lr(trainer.step, cfg["max_steps"], cfg.get("warmup_steps", 0),
                       cfg["lr_max"], cfg["lr_min"])
        row = {
            "step": trainer.step, "tokens_seen": trainer.tokens_seen,
            "loss": loss, "lr": lr,
            "tok_s": trainer.tokens_seen / max(time.perf_counter() - t0, 1e-9),
            "block_size": trainer.collator.block_size,
            **{k: round(v, 3) for k, v in getattr(trainer, "last_terms", {}).items()},
            "oput": trainer._oput_due(),
        }
        metrics.write(json.dumps(row) + "\n")
        if trainer.step % 50 == 0:
            metrics.flush()
            print(json.dumps(row), flush=True)
        if trainer.tokens_seen - last_ckpt >= cfg.get("checkpoint_interval_tokens", 75_000_000):
            trainer.save_checkpoint(out / f"ckpt-{trainer.tokens_seen}")
            # each bundle is ~6.5 GB: keep the two newest or the disk fills
            bundles = sorted(out.glob("ckpt-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
            for old in bundles[:-2]:
                for f in old.glob("*"):
                    f.unlink()
                old.rmdir()
            last_ckpt = trainer.tokens_seen

    trainer.save_checkpoint(out / "final-ckpt")
    # plain weights for the exporter: `merge_mixer_state` reads a state dict,
    # not the resumable checkpoint bundle
    torch.save(trainer.model.state_dict(), out / "final-weights.pt")
    print(f"done: {trainer.tokens_seen} tokens, {trainer.step} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
