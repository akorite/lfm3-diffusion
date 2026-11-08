# Phase 0 / Phase 1 run record

Status: in progress. Numbers below are filled from the live run.

## Budget and sizing

Balance when the run started: **216.79 CU** after the gate experiments.

Teacher throughput measured on the G4 (`divelab/opdlm_train_data`, greedy
rollout, `max_new_tokens=128`):

| rollout batch | cached tokens/s |
|---|---|
| 1 (per-example loop) | 68 |
| 16 | 444 |
| 32–64 | ~700 |

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
