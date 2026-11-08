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
| Confidence threshold `tau` | 0.90 (default); 0.30 lifts throughput at a large quality cost — see below |
| Remask stride | 4–7 (adaptive) |

## Quantization formats

| Checkpoint | Format | Size | Notes |
|------------|--------|------|-------|
| `LFM3-1.7B-Diffusion` | BF16 | 3.4 GB | **Shipped.** `export.tar.gz`, md5 `34391cc0c44a44f7c6b28ba17b9d0981f` |
| `LFM3-1.7B-Diffusion-FP8` _(recipe only, not built in this run)_ | compressed-tensors `FP8_DYNAMIC` (1D per-channel W8 + dynamic per-row A8) on `feed_forward.*` + `self_attn.{q,k,v,o}_proj`; convs, embed/lm_head, norms BF16 | 2.21 GB | `scaled_mm` native |
| `LFM3-1.7B-Diffusion-NVFP4` _(recipe only, not built in this run)_ | `U8 + F8_E4M3` MLP weights (base `LiquidAI/LFM3-1.7B-NVFP4`), MCS-recalibrated `F32 input_scale`, BF16 mixers/norms/embed | 1.83 GB | W4A16 GEMV, `M <= 2048`, BF16 KV |

## Performance (measured on the shipped BF16 checkpoint)

64 tokens, one prompt, A100-class GPU, `scripts/speed_wallclock.py`.

| Mode | tokens/forward | tok/s | vs plain AR | parity vs AR greedy | text |
|------|----------------|-------|-------------|---------------------|------|
| plain AR greedy | 1.00 | 32.2 | 1.00x | — | coherent |
| **`ar_trust`, width 4** | **1.38** | **61.9** | **1.93x** | **1.000** | coherent |
| `spd`, block 8 / small 4, tau 0.90 | 1.73 | 94.0 | 2.92x | 0.167 | collapses |
| `spd`, block 32 / small 32, tau 0.30 | 9.14 | 195.3 | 6.06x | 0.042 | collapses |

`ar_trust` is the usable configuration: exact AR-greedy parity and 1.93x, with
quality unchanged (see Evaluation). The faster `spd` settings commit many
tokens per forward and then produce repetition collapse at every commit rate
tested, including the mildest — that is a denoiser-quality limit of this
checkpoint, not a decode setting, so those rows are throughput measurements,
not shippable modes.

FP8 and NVFP4 exports are not part of this checkpoint; only BF16 ships.

## Usage

```python
from lfm3_dlm.sample import generate

# the usable mode: exact AR-greedy parity, 1.93x over plain AR
out = generate(model, prompt, mode="ar_trust", max_new_tokens=64, isd_width=4)

# throughput experiments only: fast, but the text degenerates on this checkpoint
out = generate(model, prompt, mode="spd", block_size=8, small_block=4,
               hard_commit=0.90, max_new_tokens=64)
```

## Evaluation

Measured on the shipped checkpoint, base model scored identically for comparison.

| metric | base `LiquidAI/LFM3-1.7B` | this checkpoint |
|---|---|---|
| arc_easy acc (lm-eval, n=400) | 0.420 | **0.533** |
| piqa acc | 0.590 | **0.675** |
| hellaswag acc | 0.405 | 0.398 |
| causal next-token ppl (31 rows) | 4.419 | **4.396** |
| held-out diffusion loss (31 rows) | 15.717 | **4.779** |

`ar_trust` decode reproduces AR greedy exactly, so decoding costs no accuracy;
the benchmark movement above is the conversion's own effect and sits inside or
above the ~0.025 standard error at this sample size. No per-family benchmark
sweep (math/code/IFEval/BFCL) was run for this checkpoint.


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

This checkpoint is a **budget-sized** conversion, not the plan's full run:

| | plan | shipped |
|---|---|---|
| teacher cache | 150M tokens | 20M tokens |
| student corpus tokens | 300M (18,300 steps) | **25M tokens (2,704 steps)** |
| training data | mixed KTD3 blend | `divelab/opdlm_train_data` `chat` rows that fit one packed row |
| teacher signal | Top-16 cache | Top-16 cache |
| frozen weights | MLPs + non-mask embedding rows | same |
| compute | ~46–55 CU | **~94 CU** |

The 25M-token checkpoint is the delivered one because it is the best measured:
training to 50M tokens regressed both axes (ar_trust TPF 1.38 -> 1.20, held-out
4.779 -> 4.816), consistent with the literature's scale requirement for
AR-to-diffusion conversion (30-70B tokens for a 7B model)."/

Token parity is measured, not claimed: see `docs/runs/g4_phase0_phase1.md` and
`scripts/parity_sweep.py`. `ar_trust` decode is exact by construction; `spd`
parity depends on `hard_commit`.
