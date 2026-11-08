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

## The prompt-format bug (found mid-run, fixed, run restarted)

The vendored tokenizer ships no `chat_template`, and the first run fed prompts
as raw text. On the base model the difference is decisive:

| prompt form | base model output |
|---|---|
| raw text `How do I stop Quora ...` | `Quora does not send me emails.` ×5 (degenerate repetition) |
| `<\|startoftext\|>user\n…<\|im_end\|>\n<\|startoftext\|>assistant\n` | a coherent `<think>` answer about Quora's notification settings |

`lfm3_dlm.tokenizer.format_chat` now spells the format out, and every prompt
path uses it: cache builder, corpus reader, eval, held-out eval, parity sweep.
The first run was stopped and the cache rebuilt, because training on raw-text
prompts teaches the model to continue raw text — exactly the failure the base
model shows.

**Restarted run (final):** cache `cache3` (8M tokens, capped at one packed row
per answer), training `run4` (7,000 steps), same frozen-MLP / mask-row-only
configuration, same supervisor-style endgame. The first run's `cache2`/`run3`
artifacts are kept only for comparison.

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

## Live status (VM clock 16:38 UTC, session up 3h22m)

- training step **2,000 / 7,000** (29%), loss deciles 59.5 → 27.8 (plateau band 27-38, block-32 noise)
- throughput **5,393 physical tok/s** after killing a duplicate trainer that
  was sharing the GPU (the interleaved step numbers were the tell)
- one checkpoint on disk; balance **183.32 CU** (floor 50)
- projected finish ~21:00 UTC, Colab cutoff 01:16 UTC

**Endgame pre-flight:** all ten run scripts import cleanly
(`spd_sweep`, `parity_sweep`, `heldout_eval`, `eval_trained`, `finish_run`,
`summarize_run`, `export_trained`, `train_g4`, `build_teacher_cache`,
`bench_training_step`), so the final stage should fail on measurements rather
than on an import typo.

## Why SPD cannot meet a parity bar (measured, tiny model + real model)

Same experiment twice — tiny test model and the real 1.7B — gives the same
shape: SPD's first emitted token matches AR greedy, then it diverges at the
second and prefix parity collapses to ~1/16.

| model / checkpoint | config | parity | TPF |
|---|---|---|---|
| **base `LiquidAI/LFM3-1.7B`, untouched** | ar_trust | 1.000 | **1.07** |
| tiny (random weights) | spd, block 4, hc 0.90/0.99/1.01 | 0.062 | 1.14 |
| tiny (random weights) | spd, block 8, hc 0.90/0.99/1.01 | 0.062 | 1.45 |
| **real 1.7B, ckpt-25004750** | spd | 0.062 | 1.45–1.60 |
| **real 1.7B, ckpt-4003234** | spd | **0.188 / 0.250** | 1.33–1.60 |
| **real 1.7B, ckpt-4003234** | ar_trust, width 4 or 8 | **1.000** | **1.07** |

The base model sets the floor: it decodes at 1.07 tokens per forward under the
same harness, so a trained TPF of 1.07 means no gain at all and 1.38 is a real
+29% over the baseline.

The real-model SPD number moves with training (0.062 → 0.19–0.25), so the
"structurally zero" reading is too strong; what holds is that SPD stays far
from 0.95 while AR-Trust sits at 1.000.

**Draft quality, not the verifier (measured, ckpt-4003234 = the 4M-token checkpoint).** Same prompt, same
16 tokens, same verifier — only the draft source changes:

| draft source | parity | TPF | forwards for 16 tokens |
|---|---|---|---|
| model's own mask-draft | 1.000 | **1.07** | 15 |
| oracle draft (AR continuation) | 1.000 | **2.29** | 7 |

So the verify/commit machinery reaches 2.29 tokens per forward — above the 1.5
bar — and the model's own proposals are simply not accepted yet. This is a
draft-quality gap, not a draft-path bug: the fix is a model that predicts
masked-block tokens better, which is exactly what the running training buys.

**Why the drafts are rejected (measured at ckpt-4003234).** Ask the model to fill
a fully masked block from the prompt alone — the exact conditioning the sampler's
mask-draft uses on its first step — and compare its argmax with the AR
continuation:

| metric | value |
|---|---|
| positions compared | 24 |
| argmax agrees with AR | **2 (0.083)** |
| model's first six fills | `124901, 278, 278, 278, 278, 278` |

Repeated on three prompts with the block partially revealed (the way training
draws it):

| conditioning | prompt 1 align | prompt 2 align | distinct proposals |
|---|---|---|---|
| all MASK | 0.125 | 0.062 | 2–5 |
| 2 tokens revealed | 0.143 | 0.071 | 5 |
| 4 tokens revealed | **0.417** | 0.083 | 6 |

So the all-mask collapse is partly an out-of-distribution artefact — seeding the
block raises alignment on one prompt by 3.3x and restores varied proposals — but
even seeded it stays far from usable, and the second prompt barely moves. The
decode-side lever (seed the draft block instead of masking it entirely) is real
but not sufficient on its own.

The model collapses the masked block onto one repeated token, so a draft is
almost never the token the verifier accepts. Two candidate causes, both real:
training draws `t ~ U[0.35, 0.95]`, so an **all-masked** block is at the edge of
what the collator ever shows it; and the checkpoint is 2,441 of 7,000 steps in.

**The speed criterion is the one at risk.** On ckpt-4003234, ar_trust TPF is
**1.07 for both width 4 and width 8** — width changes nothing, which means
essentially zero drafts are accepted beyond the first token. AR-Trust is
currently decoding at AR speed with verification overhead.

`hard_commit` changes nothing, which is the tell: SPD commits a position when
the *noisy-block* predictor is confident, and that predictor is conditioned on
a partially denoised block, not on the causal prefix AR greedy uses. It is a
different decoding algorithm, not a faster AR — so "SPD at 95% AR parity" is
not a tuning problem, it is the wrong frame.

**Caveat on that evidence.** The tiny-model arm is weak on its own — a
random-weight model's AR reference is arbitrary, so any decode path diverges
from it. The real 1.7B measurement at the 25M-token checkpoint is the stronger
data point (parity 0.062, TPF 1.45–1.60 at isd_width 4), and it agrees. But
that checkpoint was ~2,700 of 7,000 steps in, so a converged model could commit
the AR argmax far more often by chance. The endgame sweep over
`hard_commit` × block size is what decides; this note records the prior, not a
verdict.

The parity-carrying path is AR-Trust: drafts are accepted only where the
model's own top-1 agrees, so parity is 1.000 by construction. Its speed is set
by **draft acceptance**, which is why the endgame sweeps `isd_width` 2/4/8
rather than any SPD knob.


## Full battery at the 25M-token checkpoint (dry run of the endgame)

Run against `ckpt-25004750` (~2,700 of 7,000 steps) to validate every endgame
script before the final weights exist. Values are the harnesses' own output.

| criterion | measurement | value |
|---|---|---|
| 1 parity | `parity_sweep.py`, 4 prompts x 24 tokens, ar_trust mean | **1.000** (worst 1.000) |
| 2 speed | same sweep, mean TPF | ar_trust_2 1.27, **ar_trust_4 1.38**, ar_trust_8 1.38 |
| 3 quality | `heldout_eval.py`, 31 rows that fit | trained **4.779** vs base **15.717** |
| 4 tests | `pytest tests/ -q` | 98 passed |
| 5 export | `finish_run.py` export step | rc=0, samples coherent |

Per-config parity/TPF, straight from `parity.md`:

| config | mean parity | worst parity | mean TPF |
|---|---|---|---|
| ar_trust_2 | 1.000 | 1.000 | 1.27 |
| ar_trust_4 | 1.000 | 1.000 | 1.38 |
| ar_trust_8 | 1.000 | 1.000 | 1.38 |
| spd_0.90 | 0.167 | 0.042 | 1.60 |
| spd_0.99 | 0.198 | 0.042 | 1.55 |
| hybrid_auto | 0.167 | 0.042 | 1.60 |

Two readings. Parity is exact and stable, so criterion 1 is already satisfied.
Speed is the open one: 1.38 against a 1.5 bar, but it moved 1.07 -> 1.38 as the
model went from 4M to 25M tokens, which is the direction that matters with
4,300 steps still to run.


## Acceptance self-distillation: two formulations, both negative

The throughput lever is draft acceptance, so `scripts/rl_accept.py` optimises it
directly, expert-iteration style: the model's own greedy continuation is the
target, the masked-block prediction is the policy, cross-entropy on the masked
slots is the signal. Both phases warm-started from `ckpt-25004750` (TPF 1.38)
and ran 1,500 steps at lr 1.5e-5, frozen MLPs.

| row construction | in-batch mask agreement | ar_trust parity | ar_trust TPF |
|---|---|---|---|
| start (`ckpt-25004750`) | — | 1.000 | **1.38** |
| prompt + noisy-stream MASK x4, bidirectional | 0.56-0.81 | 1.000 | 1.06-1.09 |
| prompt + pending token + causal MASK x4 (mirrors the draft feed) | 0.58-1.00 | **0.833** | 1.17-1.20 |

Both made the real criterion worse while the training metric improved, and the
second broke parity (near-tie flips under bfloat16 once the model is pushed off
its supervised optimum). The conclusion is not "RL does not work"; it is that
optimising agreement on a proxy row destroys the calibration the verifier
depends on. Neither artifact is used: `run5/rl1500.pt` and `run5/rl_draft.pt`
are recorded and discarded, and the pre-RL weights remain the best model.

The lever that demonstrably moved TPF is supervised scale: 1.07 at 4M tokens to
1.38 at 25M. That is where the remaining compute goes.


## Scaling past the 7,000-step override

`configs/g4_run.yaml` budgets 18,300 steps (150M corpus tokens); the run was
launched with a `--max-steps 7000` override as a first leg. The RL phases above
were the wrong lever, so the remaining compute goes to restoring that schedule
rather than inventing a new one: resume at `--max-steps 18300`, which keeps the
designed cosine shape instead of bolting a second anneal onto a finished one.

Evidence for this being the lever: ar_trust TPF 1.07 at 4M tokens, 1.38 at 25M,
against a base-model floor of 1.07. A separate acceptance post-phase was tried
and reverted (above), so the schedule is the only change.

An unattended driver (`/content/vm_driver.py`) resumes chunks until step 18,300
and then runs `scripts/endgame.sh` on its own, so the battery fires the moment
training finishes instead of waiting on a polling loop.

## Post-training deliverables

When training finishes the chain produces, in order:

1. `/content/run3/final-ckpt` (resumable) and `/content/run3/final-weights.pt`
2. `/content/export` — BF16 HF directory via `merge_mixer_state`, with the
   **trained** mask row and untouched frozen MLPs
3. `/content/eval.json` — three generations with their TPF, the number of
   weight keys that loaded, and a held-out loss computed over cache rows

**Export pre-flight (verified on the 25M-token checkpoint):** the merge
produces a 24-layer HF directory that loads with `Lfm3ForCausalLM`, and the
trained mask row is *not* re-seeded from the pad row
(`mask_norm 0.434` vs `pad_norm 0.436`, `allclose == False`) — the bug a
pre-flight review caught before the run would have shipped it.

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
