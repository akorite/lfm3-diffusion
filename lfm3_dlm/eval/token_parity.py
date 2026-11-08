"""Token-parity simulation against AR ground-truth continuations.

Replays AR reference continuations through the dual-mode sampler via the
`forced_tokens` / `draft_fn` seams: AR-Trust teacher-forced verification at
ISD strides 4..7 and an SPD confidence-threshold sweep. Reports acceptance
and parity metrics plus predicted vs realized tokens-per-forward (TPF) and
tok/s speedup vs the AR baseline (1 token/forward).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

SPD_TAU_SWEEP = (0.30, 0.55, 0.80, 0.95)   # confidence-commit thresholds
ISD_STRIDES = (4, 5, 6, 7)


@dataclass
class ParityReport:
    mode: str
    stride: int = 0
    tau: float = 0.0
    n_tokens: int = 0
    n_forwards: int = 0
    token_parity: float = 0.0          # fraction of emitted == AR continuation
    mean_accepted_len: float = 0.0     # tokens committed per verify round
    exact_match_steps: int = 0         # consecutive correct positions at step
    top1_flip_rate: float = 0.0        # fraction of positions where top-1 differs
    kl_vs_ar: float = 0.0              # mean KL(AR || sampler) over emitted positions
    acceptance_efficiency: float = 0.0 # mean_accepted_len / stride
    predicted_tpf: float = 0.0         # analytical tokens/forward
    realized_tpf: float = 0.0          # emitted / n_forwards from the sampler
    expected_tpf: float = 0.0          # emitted / num_steps (caller-computed)
    tok_per_s: float = 0.0
    speedup_vs_ar: float = 0.0         # realized_tpf / 1.0
    details: dict = field(default_factory=dict)


def _parity_prefix(emitted: list[int], reference: list[int]) -> int:
    """Number of exact-match positions before the first divergence."""
    n = 0
    for a, b in zip(emitted, reference):
        if a != b:
            break
        n += 1
    return n


def _collect_step_stats(generate_fn, model, prompt, continuation, mode,
                        isd_width, draft_fn=None):
    """One replay pass that also captures per-step predictors for KL/flip stats.

    The step hook observes `ar_verify`/`spd` phases; we pull the row's
    predictors at emitted positions to compare sampler logits vs the AR
    reference continuation.
    """
    accepted_per_round: list[int] = []
    seen_emitted = [0]
    predictor_rows: list[torch.Tensor] = []

    def hook(r, phase, feeds, row):
        delta = len(row.emitted) - seen_emitted[0]
        if delta > 0:
            accepted_per_round.append(delta)
            seen_emitted[0] = len(row.emitted)
        for pos in list(row.predictors):
            if row.predictors[pos] is not None:
                predictor_rows.append(row.predictors[pos])

    t0 = time.perf_counter()
    res = generate_fn(model, prompt, mode=mode,
                      max_new_tokens=len(continuation),
                      isd_width=isd_width, draft_fn=draft_fn,
                      forced_tokens=list(continuation),
                      eos_token_ids=(-1,))   # never terminate early
    elapsed = time.perf_counter() - t0
    emitted = res.ids[0, prompt.shape[1]:].tolist()[: len(continuation)]
    return res, emitted, accepted_per_round, predictor_rows, elapsed


def _kl_and_flips(predictors: list[torch.Tensor],
                  continuation: list[int],
                  ar_logits_fn=None) -> tuple[float, float]:
    """Top-1 flip rate and mean KL vs AR reference logits.

    When `ar_logits_fn` is None the continuation itself is the reference: a
    position flips if the sampler's top-1 != the AR token; KL is estimated
    against a one-hot AR distribution at the reference token (i.e. NLL of
    the reference token).
    """
    flips = 0
    kl = 0.0
    n = min(len(continuation), len(predictors))
    for i in range(n):
        logits = predictors[i].float()
        top1 = int(logits.argmax())
        if top1 != continuation[i]:
            flips += 1
        log_p = F.log_softmax(logits, dim=-1)
        kl += float(-log_p[continuation[i]])   # KL(one-hot(ref) || p)
    return (kl / n if n else 0.0, flips / n if n else 0.0)


def simulate_parity(model, generate_fn, prompt: torch.Tensor,
                    continuation: list[int],
                    strides: tuple = ISD_STRIDES,
                    taus: tuple = SPD_TAU_SWEEP) -> dict:
    """Replay `continuation` (AR ground truth) through the sampler.

    Returns `{"ar_trust": [ParityReport per stride],
              "spd": [ParityReport per tau]}`.
    """
    reports: dict[str, list[ParityReport]] = {"ar_trust": [], "spd": []}

    # AR-Trust teacher-forced verification at each ISD stride. The oracle
    # draft proposes the AR continuation itself, modelling a perfect draft.
    for stride in strides:
        def oracle(r, gen_pos, emitted, _s=stride):
            return continuation[gen_pos: gen_pos + _s]

        res, emitted, acc, preds, elapsed = _collect_step_stats(
            generate_fn, model, prompt, continuation,
            mode="ar_trust", isd_width=stride, draft_fn=oracle)
        kl, flip = _kl_and_flips(preds, continuation)
        n = len(emitted)
        rep = ParityReport(mode="ar_trust", stride=stride, n_tokens=n,
                           n_forwards=res.n_forwards)
        rep.token_parity = _parity_prefix(emitted, continuation) / max(n, 1)
        rep.mean_accepted_len = (sum(acc) / len(acc)) if acc else 0.0
        rep.exact_match_steps = _parity_prefix(emitted, continuation)
        rep.top1_flip_rate = flip
        rep.kl_vs_ar = kl
        rep.acceptance_efficiency = rep.mean_accepted_len / stride
        # each verify round commits <= stride+1 tokens in exactly 2 forwards
        rep.predicted_tpf = (rep.mean_accepted_len or 1.0) / 2.0
        rep.realized_tpf = res.tpf
        # analytical prediction from the acceptance statistics, so the
        # comparison against realized TPF measures agreement, not identity
        rep.expected_tpf = rep.predicted_tpf
        rep.tok_per_s = n / max(elapsed, 1e-9)
        rep.speedup_vs_ar = rep.realized_tpf / 1.0
        reports["ar_trust"].append(rep)

    # SPD replay at swept confidence thresholds.
    for tau in taus:
        res, emitted, acc, preds, elapsed = _collect_step_stats(
            generate_fn, model, prompt, continuation,
            mode="spd", isd_width=6)
        kl, flip = _kl_and_flips(preds, continuation)
        n = len(emitted)
        rep = ParityReport(mode="spd", tau=tau, n_tokens=n,
                           n_forwards=res.n_forwards)
        rep.token_parity = _parity_prefix(emitted, continuation) / max(n, 1)
        rep.mean_accepted_len = (sum(acc) / len(acc)) if acc else 0.0
        rep.exact_match_steps = _parity_prefix(emitted, continuation)
        rep.top1_flip_rate = flip
        rep.kl_vs_ar = kl
        rep.predicted_tpf = rep.mean_accepted_len
        rep.realized_tpf = res.tpf
        # analytical prediction from the acceptance statistics, so the
        # comparison against realized TPF measures agreement, not identity
        rep.expected_tpf = rep.predicted_tpf
        rep.tok_per_s = n / max(elapsed, 1e-9)
        rep.speedup_vs_ar = rep.realized_tpf / 1.0
        reports["spd"].append(rep)

    return reports
