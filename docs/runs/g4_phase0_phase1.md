# Phase 0 / Phase 1 run record

Status: in progress. Numbers below are filled from the live run.

## Budget and sizing

Balance when the run started: **216.79 CU** after the gate experiments.

Teacher throughput measured on the G4. Rollout mode (teacher generates) is
decode-bound and too slow to reach the plan's volume:

| rollout batch | cached tokens/s |
|---|---|
| 1 (per-example loop) | 68 |
| 16 | 444 |
| 32–64 | ~700 |

Reference forcing over a long-answer corpus is the path that fits the budget.
`nvidia/Llama-Nemotron-Post-Training-Dataset` (`SFT`, `chat`; `input` →
`output`) at `seq_len=4096`, batch 16, with **length-bucketed batches** and
prompt-tail truncation:

| builder | cached tokens/s |
|---|---|
| per-example, no bucketing (opdlm) | ~200 |
| batched, no bucketing | ~840 |
| batched + length bucketing (Nemotron) | **17,047** |

The bucketing is what matters: an unbucketed chat batch pads every row to the
longest one, and the padding work dominates.

At 700 tok/s a 6M-token cache costs ~2.4 h (~24 CU). Phase 1 at 5.1k corpus
tok/s for 12,200 steps (100M corpus tokens) costs ~5.4 h (~54 CU). Export and
eval add ~0.5 h. Planned total ~8.5 h ≈ 85 CU, leaving >120 CU of headroom.

Deliberate deviation from the plan's 150M teacher / 300M corpus: those need
~94 h of teacher rollout alone at the measured rate, which the balance cannot
carry. The run is sized to finish rather than to match the plan's token count.

## Run shape (final)

The pipeline runs as one detached chain on the G4 so a scheduler hiccup does
not stall it:

```
build_teacher_cache.py --data-files /content/data/chat.jsonl ... --max-tokens 20M
  && train_g4.py --cache /content/cache2 ... --max-steps 12200 --resume
```

Colab resets the VM at 12 h, and Google Drive mount is unavailable on this
account, so the run is sized to finish inside one session (cache ~20 min at the
measured 17k tok/s, training ~5 h at 5.1k corpus tok/s). Checkpoints are pruned
to the two newest so the ~6.5 GB bundles cannot fill the disk.

A VM-side supervisor (`supervisor.sh`) watches the step counter: if the trainer
dies it relaunches with `--resume` (capped at three retries), and when the
target is reached it runs `finish_run.py` (export -> eval -> parity), tars the
export, and writes `/content/FINISHED`. The whole endgame therefore completes
without a live operator.

## Phase 0 — teacher cache

- Teacher: `LiquidAI/LFM3-1.7B`, FP8 off (measured neutral), rollout mode.
- Corpus: `divelab/opdlm_train_data` (`question` → teacher rollout).
- Command:

```
python scripts/build_teacher_cache.py --dataset divelab/opdlm_train_data \
  --prompt-field question --response-field ground_truth_answer \
  --generate --max-new-tokens 256 --seq-len 1024 \
  --out /content/cache --max-tokens <target>
```

- Measured rate: ~4.7k tokens/s in the live run (17k in the short benchmark;
  the live mix includes shorter rows and the head cast per chunk)
- Tokens cached: 20M target from `nvidia/Llama-Nemotron-Post-Training-Dataset`
  `SFT/chat`, local JSONL, rows filtered to `P+(1+K)R <= 4096`

## Parity gate

`simulate_parity` over held-out AR continuations; Phase 1 launches only if the
predicted SPD TPF ≥ 4.5. Result: _pending_

## Phase 1 — training

```
python scripts/train_g4.py --cache /content/cache --dataset divelab/opdlm_train_data \
  --prompt-field question --response-field ground_truth_answer \
  --config configs/g4_300m.yaml --out /content/run --max-steps <n>
```

- Measured throughput: 5.4k corpus tok/s (gate 22k — not met; recorded)
- Steps / tokens: 12,200 steps, ~9.0k physical (~4.5k corpus) tokens per step
  (OPUT disabled for this run)
- Live throughput: ~7.2k physical tokens/s, so ~5.5 h for the schedule
- Loss trajectory (mean per decile of steps, first 1,050 steps):
  70.1 → 43.8 → 39.1 → 36.1 → 33.5 → 32.1 → 32.6 → 30.4 → 30.5 → 29.6.
  Per-step variance is large (4 documents per step, fresh mask draw each step,
  and block size annealing 4→32 during warmup), so the decile means are the
  signal, not any single step.

## Export and eval

- `merge_mixer_state` → BF16 export; `MODEL_CARD.md` updated with the honest
  numbers.
- Held-out loss / sample: _pending_

## Token parity (`>95%` goal)

Two decode paths exist and they trade speed for exactness:

| path | parity vs AR greedy | how |
|---|---|---|
| `ar_trust` | exact by construction | every drafted block is verified against the model's own top-1 before commits |
| `spd` | measured, threshold-dependent | parallel commits at confidence `> hard_commit` |
| `hybrid_auto` | `ar_trust` inside `<tool_call>`, `spd` elsewhere | routing on the tool-call delimiters |

`scripts/parity_sweep.py` measures the real number (no forced tokens: the
sampler runs freely and its emitted prefix is compared to the AR greedy
reference). `hard_commit` is now a `generate(..., hard_commit=...)` knob, so the
SPD parity/speed tradeoff is tunable per call rather than baked into a constant.

Expected shape: `ar_trust` at 1.000 parity with TPF ~2–3; `spd` at high TPF with
parity that rises as `hard_commit` → 1.0 (a threshold of 1.0 degenerates to
one-token-at-a-time commits, i.e. AR). The sweep records all rows, so the
"≥95% parity at the best TPF" point is read off the table rather than assumed.


## Pre-flight review fixes (before the first training step)

An unattended 10-hour run was reviewed first; every finding that would have
wasted it was fixed and committed:

| Finding | Fix |
|---|---|
| Cache accepted `P+R <= seq_len` but the collator needs `P+(1+K)R`, so long rows raise on the first draw | builder enforces the packing bound; the reader also skips rows that do not fit, so an older cache stays usable |
| Driver ignored stage return codes and always wrote a success marker | every stage is checked; failures abort with an error record |
| `rc` unbound when phase 0 was skipped on resume | initialised |
| LR never annealed (config `max_steps` ≠ CLI `max_steps`) | the CLI value drives the cosine |
| Resume picked the lexicographically last checkpoint | numeric sort |
| Export re-seeded the mask row from the pad row, discarding the only embedding row training moves | merge no longer calls it |
| OPUT built an unchunked `[positions, 125033]` fp32 block per example | chunked, shared with `sliced_lse` |
| Checkpoints ~6.5 GB each, nothing pruned | keep the two newest |
| Parity gate's rc discarded and it scored AR-Trust instead of the documented SPD criterion | both fixed |
| `--max-steps` break unreachable because `tokens_seen` counts physical lanes | break on steps; token count is a guard only |
| Eval loaded weights with `strict=False` and never scored the checkpoint | records how many keys loaded |

## Early parity probe (checkpoint at 25M physical tokens, ~step 2,700)

Two prompts x 16 tokens, block 8 / small 4, measured against AR greedy with no
forced tokens:

| config | parity | TPF |
|---|---|---|
| `ar_trust` (isd_width 4) | **1.000** | 1.23 |
| `spd`, hard_commit 0.90 | 0.062 | 1.60 |
| `spd`, hard_commit 0.99 | 0.062 | 1.45 |

Two things this settles early:

- **The >95% bar is met today on the verified path.** AR-Trust is exact by
  construction, and the measurement agrees (1.000 on both prompts) — it does
  not depend on how converged the student is.
- **SPD is not usable yet.** 6% parity at 1.5 TPF is worse than plain AR on
  both axes. Raising `hard_commit` changed nothing, because at block 8 the
  two-step top-1 rule and the block-end bonus fire before the threshold does.
  SPD needs the converged checkpoint and a block-size sweep before it is worth
  offering as the fast path.

## Post-training deliverables

When training finishes the chain produces, in order:

1. `/content/run3/final-ckpt` (resumable) and `/content/run3/final-weights.pt`
2. `/content/export` — BF16 HF directory via `merge_mixer_state`, with the
   **trained** mask row and untouched frozen MLPs
3. `/content/eval.json` — three generations with their TPF, the number of
   weight keys that loaded, and a held-out loss computed over cache rows

**Honest limit on that loss:** the Nemotron chat split ships as a single file,
so the "held-out" rows come from the same distribution the model trained on —
it measures fit, not generalisation. The generation samples and the AR-parity
number are the claims that do not depend on that split.

### Why `ar_trust` is exact

`_run_ar` drafts `k` candidates, verifies them with one causal forward, and
accepts a prefix only while the model's own argmax agrees
(`predictors[C+j].argmax() == cand[j]`); the first mismatch is committed from
`predictors[C+m].argmax()`, which is exactly what AR greedy would emit. So the
emitted sequence equals uncached AR decoding token-for-token by construction,
and `tests/test_dual_sampler.py` pins that equality. SPD is the fast path and
its parity is an empirical number.

The parity claim is measured separately, on the trained weights:

```
python scripts/parity_sweep.py --weights /content/run3/final-weights.pt \
  --prompts 6 --tokens 32 --out /content/parity.md
```

`ar_trust` is exact by construction; `spd` rows are read off the same table at
`hard_commit` 0.90 and 0.99. The shipped decode recommendation is whichever row
first reaches ≥95% parity with the best TPF — if `spd` never gets there,
`ar_trust` (or `hybrid_auto`, which routes tool-call spans through it) is the
default and the SPD tradeoff is documented rather than hidden.
