"""Locked benchmark runner: pinned tasks/decode configs, guarded metrics."""
from __future__ import annotations

import time
from typing import Callable, Iterable

import torch

from .settings import BENCHMARK, DECODE_CONFIGS, METRIC_FIELDS, report_metric


def token_global_mean(per_sequence_losses: Iterable[torch.Tensor]) -> float:
    """Token-global loss averaging: weight each sequence by its token count.

    Per-token mean over the whole eval set (not a mean of sequence means).
    """
    num = den = 0.0
    for loss in per_sequence_losses:
        n = loss.numel() if loss.ndim else int(loss)
        num += float(loss.sum()) if loss.ndim else float(loss) * n
        den += n
    return num / max(den, 1)


def _check_decode(decode: dict) -> dict:
    missing = [f for f in ("mode", "B", "small_block") if f not in decode]
    if missing:
        raise ValueError(f"decode config missing fields: {missing}")
    modes = {c["mode"] for c in DECODE_CONFIGS}
    if decode["mode"] not in modes:
        raise ValueError(f"decode mode {decode['mode']!r} is not a pinned mode "
                         f"{sorted(modes)}")
    return dict(decode)


def run_eval(model, generate_fn: Callable, prompts: dict[str, torch.Tensor],
             decode: dict | None = None,
             scorer: Callable | None = None,
             max_new_tokens: int = 64) -> dict:
    """Run the pinned benchmark slice over `prompts` {task_name: token ids}.

    `scorer(task, ids_tensor, result)` returns a task score (default: 1.0 —
    generation completed). Every returned metric row carries the pinned
    decode metadata `(mode, B, small_block, tau_dec, tau_acc, TPF, tok/s,
    peak_VRAM)`.
    """
    decode = _check_decode(decode or DECODE_CONFIGS[0])
    unknown = set(prompts) - set(BENCHMARK["tasks"])
    if unknown:
        raise ValueError(f"tasks outside the pinned benchmark: {sorted(unknown)}")

    rows = []
    for task, ids in prompts.items():
        t0 = time.perf_counter()
        res = generate_fn(model, ids, mode=decode["mode"],
                          max_new_tokens=max_new_tokens,
                          block_size=decode["B"],
                          small_block=decode["small_block"],
                          isd_width=decode.get("isd_width", 6))
        elapsed = time.perf_counter() - t0
        emitted = res.ids.shape[1] - ids.shape[1]
        meta = {
            "mode": decode["mode"],
            "B": decode["B"],
            "small_block": decode["small_block"],
            "tau_dec": decode.get("tau_dec", 0.55),
            "tau_acc": decode.get("tau_acc", 0.90),
            "TPF": res.tpf,
            "tok_per_s": emitted / max(elapsed, 1e-9),
            "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30
                             if torch.cuda.is_available() else 0.0),
        }
        assert set(meta) == set(METRIC_FIELDS)
        score = scorer(task, ids, res) if scorer else 1.0
        rows.append(report_metric(f"{task}.score", score, decode=meta))
    return {"decode": decode, "metrics": rows}
