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

## Stopping

```bash
colab stop -s lfm3-g4
```

Never touch an operator's existing session.
