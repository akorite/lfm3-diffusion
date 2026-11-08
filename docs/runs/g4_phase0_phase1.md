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

- Measured rate: _pending_
- Tokens cached: _pending_

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
- Steps / tokens: _pending_
- Final loss: _pending_

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
