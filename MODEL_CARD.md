---
license: other
license_name: lfm-open-license-v1.0
license_link: https://www.liquid.ai/lfm-license
base_model: LiquidAI/LFM3-1.7B
tags:
  - diffusion
  - text-generation
  - block-diffusion
pipeline_tag: text-generation
---

# LFM3-1.7B-Diffusion

**LFM3-1.7B-Diffusion** is a block-diffusion language model derived from
[`LiquidAI/LFM3-1.7B`](https://huggingface.co/LiquidAI/LFM3-1.7B) by
self-distillation conversion. It generates text via **parallel block
diffusion**: fixed-size blocks of tokens are iteratively unmasked under a
semi-autoregressive schedule, while retaining an autoregressive-trust
(AR-Trust) verification path for lossless decoding.

- **Task:** text generation (parallel block-diffusion)
- **Base model:** `LiquidAI/LFM3-1.7B`
- **Architecture:** LFM3 hybrid — full-attention / sliding-attention /
  gated short-convolution layers; mask token id `125033`
- **License:** [LFM Open License v1.0](https://www.liquid.ai/lfm-license)

## Decode modes

| Mode | Description |
|------|-------------|
| `spd` | Speculative parallel decoding — unmask high-confidence tokens in parallel within each block |
| `ar_trust` | AR-Trust lossless verification — parallel draft, autoregressive-verified accept; output identical to AR decoding |
| `hybrid_auto` | Automatic switching: `spd` for low-risk spans, `ar_trust` when verification fails |

## Decode hyperparameters

| Parameter | Value |
|-----------|-------|
| Block size `B` | 32 |
| `small_block` | 8 |
| Confidence threshold `tau` | TODO(GPU run): fill final tau per mode |
| Remask stride | 4–7 (adaptive) |

## Quantization formats

| Checkpoint | Format | Size | Notes |
|------------|--------|------|-------|
| `LFM3-1.7B-Diffusion` | BF16 | ~3.4 GB | Reference export |
| `LFM3-1.7B-Diffusion-FP8` | compressed-tensors `FP8_DYNAMIC` (1D per-channel W8 + dynamic per-row A8) on `feed_forward.*` + `self_attn.{q,k,v,o}_proj`; convs, embed/lm_head, norms BF16 | 2.21 GB | `scaled_mm` native |
| `LFM3-1.7B-Diffusion-NVFP4` | `U8 + F8_E4M3` MLP weights (base `LiquidAI/LFM3-1.7B-NVFP4`), MCS-recalibrated `F32 input_scale`, BF16 mixers/norms/embed | 1.83 GB | W4A16 GEMV, `M <= 2048`, BF16 KV |

## Performance (TODO — fill from GPU benchmark run)

| Model | Mode | TPF | tok/s | Retention vs AR | Peak VRAM |
|-------|------|-----|-------|-----------------|-----------|
| BF16 | `spd` | TODO | TODO | TODO | TODO |
| BF16 | `ar_trust` | TODO | TODO | 100% (lossless) | TODO |
| FP8 | `spd` | TODO | TODO | TODO | TODO |
| MCS-NVFP4 | `spd` | TODO | TODO | TODO | TODO |

Targets: `TPF >= 4.5` (SPD), >= 90% benchmark retention under `spd`,
>= 95% under `ar_trust`, >= 3.5x tok/s over BF16 AR.

## Usage

```python
from lfm3_dlm.sample import generate

out = generate(model, tokenizer, prompt, mode="spd", block_size=32, small_block=8)
```

## Evaluation

See `docs/runs/final_eval.md` for the per-family benchmark record
(math, code, IFEval, BFCL/XML) across all three checkpoint formats.


## Training status

This card describes the conversion recipe. The checkpoint shipped from this
repo is produced by `scripts/run_pipeline.py` on a single G4 and its exact
numbers are recorded in `docs/runs/g4_phase0_phase1.md`:

- teacher: `LiquidAI/LFM3-1.7B` (BF16; FP8 measured neutral)
- corpus: `nvidia/Llama-Nemotron-Post-Training-Dataset`, `SFT/chat`
- cache: Top-16 teacher distributions at 68 B/token, length-bucketed batches
- student: 1.7B, frozen MLPs + mask-row-only embedding, FlexAttention block
  masks, chunked classifier LSE
- decode: `AR-Trust` verification is exact by construction; `SPD` parity is
  measured, not assumed (`scripts/parity_sweep.py`)

The token counts and losses below the recipe in this card describe the
*planned* 300M-token run; the shipped checkpoint is sized to the available
compute and the run record states the difference.
