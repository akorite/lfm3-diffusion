# G4 conversion runbook

Phase 0 caches teacher distributions; Phase 1 trains the diffusion objective.
Both run on one named Colab session so the VM survives between calls.

```bash
colab new -s lfm3-g4 --gpu G4          # RTX PRO 6000 Blackwell, 96 GB
colab exec -s lfm3-g4 -f bootstrap.py  # torch + transformers>=5.20 + repo
```

## Corpus (verified 2026-11-08)

| Dataset | Shape | Use |
|---|---|---|
| `divelab/opdlm_train_data` | one `opdlm_train.json`, 1.72 GB, 61,816 prompts: math 20,222 / code 21,594 / science 10,000 / chat 10,000 | the prompt source |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | `SFT/chat/chat.jsonl`, `RL/instruction_following/*.jsonl` | tool-call and instruction-following quotas |
| `LiquidAI/structured-extraction` | `showcase.parquet`, `test.parquet` | XML `<tool_call>` / JSON validity |

Stream it (`load_dataset(..., streaming=True)`); the single 1.72 GB JSON is
memory-heavy to load whole.

## Phase 0 — teacher cache

```bash
python scripts/build_teacher_cache.py \
  --dataset <hf-id> --prompt-field prompt --response-field response \
  --out /content/cache --max-tokens 150_000_000 --shard-tokens 4_000_000
```

Writes `shard-*.bin` (68 B/token) plus `index.jsonl`; `meta.json` records the
token count and rate, and a rerun resumes from the index instead of restarting.

## Phase 1 — training

```bash
python scripts/train_g4.py --cache /content/cache --dataset <hf-id> \
  --config configs/g4_300m.yaml --out /content/run
```

## Measuring and piloting

```bash
python scripts/bench_training_step.py --s-phys 4096 --microbatch 1 --flex   # exits 1 below the gate
python scripts/pilot_ab.py --budget-seconds 600 --flex                      # K=2 vs K=4, matched compute
```

`bench_training_step.py` is the gate: it prints corpus tok/s and peak VRAM and
returns non-zero when either misses, so a run script can be gated on it.
`pilot_ab.py` gives each K the same wall-clock budget and scores both with one
shared held-out evaluation, writing `docs/runs/g4_pilot.md`.

`n_masks` in the collator (trainer config key) selects the packing: `2` is the
complementary pair used today, `K>2` packs K independent mask copies over one
clean context.

## Gates (abort before spending Phase-1 compute when one fails)

| Gate | Threshold | Measured 2026-11-08 on G4 |
|---|---|---|
| Corpus throughput | `>= 22k` tok/s (BF16 `>= 11k` on A100) | **5.4k** at mb=1 with FlexAttention — fails |
| Peak VRAM | `< 65 GB` without activation checkpointing | **29.1 GB** at mb=1 — passes; 90 GB at mb=4 |
| Finite loss over the smoke | no NaN/Inf | passes |

Layer checkpointing: the trainer flag (`activation_checkpointing`) uses the
model's `checkpoint_layers` kwarg and is tested; the native
`gradient_checkpointing_enable` path also works and costs ~30% step time for
~4 GB, so it is preferred when memory is the binding constraint. Neither is on
by default — VRAM already clears the gate at `microbatch: 1-2`.

FlexAttention is wired and buys ~10%, not the 4x the mask cost suggested; the
step sits at ~19% of BF16 peak, so the remaining gap is utilisation, not mask
materialisation. Full numbers and the checkpointing incompatibility are in
`docs/runs/g4_gate.md`.

FP8 frozen MLPs measured **no speedup** (9.1k vs 9.3k tok/s) and more VRAM,
because the input-gradient copy keeps the bf16 weights resident; the FP8 path
is opt-in (`fp8_mlps: true`) until it earns its place.

## Long jobs on this CLI

The Colab CLI documents **activity-based liveness**: a session stays alive while
the *kernel* is active. A `setsid nohup` process is not kernel activity, so a
detached chain can be reaped mid-run even though it looked healthy — observed
here as a cache build that finished and then never started training.

Run long work as a **blocking `colab exec` with its own timeout**, in chunks:

```bash
colab exec -s lfm3-g4 -f chunk.py --timeout 3200   # chunk.py runs ~48 min, then exits
```

and keep `checkpoint_interval_tokens` small enough (4M tokens ≈ 450 steps) that
a chunk boundary loses only minutes. `colab run` is not a substitute: it tears
the VM down when the script exits.

## Stopping

```bash
colab stop -s lfm3-g4
```

Never touch an operator's existing session.
