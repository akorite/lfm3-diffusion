"""Pinned benchmark configuration and metric-sanity guardrails."""
from __future__ import annotations

# Tasks pinned for the conversion eval. `structured_extraction` covers the
# LiquidAI structured-extraction set (XML tool_call + JSON schema checks).
BENCHMARK = {
    "tasks": ("gsm8k", "math500", "humaneval_plus", "ifeval", "bfcl_xml",
              "structured_extraction"),
    "n_shots": {"gsm8k": 4, "math500": 0, "humaneval_plus": 0,
                "ifeval": 0, "bfcl_xml": 0, "structured_extraction": 0},
}

# Locked decode configurations; `B` = diffusion block size.
DECODE_CONFIGS = (
    {"mode": "spd",         "B": 32, "small_block": 8, "isd_width": 0},
    {"mode": "ar_trust",    "B": 32, "small_block": 8, "isd_width": 6},
    {"mode": "hybrid_auto", "B": 32, "small_block": 8, "isd_width": 6},
)

# Every benchmark row must carry this decode metadata.
METRIC_FIELDS = ("mode", "B", "small_block", "tau_dec", "tau_acc",
                 "TPF", "tok_per_s", "peak_vram_gb")

# Metric families that must never be compared across paradigms.
_DIFFUSION_METRICS = {"elbo", "dlm_ppl", "diffusion_nll"}
_AR_METRICS = {"ar_ppl", "ar_perplexity", "causal_ppl"}


def _family(metric: str) -> str:
    m = metric.lower()
    if m in _DIFFUSION_METRICS or "elbo" in m:
        return "diffusion"
    if m in _AR_METRICS or ("ppl" in m and "dlm" not in m):
        return "ar"
    return "other"


def compare_metrics(metric_a: str, value_a: float,
                    metric_b: str, value_b: float) -> dict:
    """Compare two metric values; refuse ELBO/PPL cross-paradigm comparisons.

    Diffusion ELBO-style bounds are not comparable to AR perplexity; mixing
    them in a leaderboard is a measurement bug, so it is a hard error.
    """
    fa, fb = _family(metric_a), _family(metric_b)
    if {fa, fb} == {"diffusion", "ar"}:
        raise ValueError(
            f"cannot compare {metric_a!r} against {metric_b!r}: "
            "diffusion ELBO bounds are not AR perplexity")
    delta = value_a - value_b
    return {"metric_a": metric_a, "metric_b": metric_b,
            "delta": delta, "rel": delta / value_b if value_b else float("inf")}


def report_metric(name: str, value: float, decode: dict | None = None,
                  context: dict | None = None) -> dict:
    """Build one benchmark metric row, enforcing decode metadata and the
    ELBO-vs-AR-PPL guardrail.

    `decode` must carry the pinned decode fields; `context` may carry a
    `baseline_metric` for paired reporting.
    """
    if decode is None or any(f not in decode for f in ("mode", "B", "small_block")):
        raise ValueError(
            "benchmark metrics require decode metadata (mode, B, small_block)")
    row = {"metric": name, "value": float(value), "decode": dict(decode)}
    ctx = context or {}
    if "baseline_metric" in ctx:
        row["comparison"] = compare_metrics(
            name, value, ctx["baseline_metric"], ctx.get("baseline_value", 0.0))
    return row
