# Final Evaluation — LFM3-1.7B-Diffusion

Run record for the tri-format benchmark pass (BF16 / PerRow-FP8 / MCS-NVFP4).
Fill every `TODO` from the GPU benchmark run; report per benchmark family —
never a blended mean.

**Run metadata**

- Date / operator: TODO
- Session / GPU: `lfm3-g4` (RTX PRO 6000 Blackwell 96GB)
- Checkpoints: BF16 / `LFM3-1.7B-Diffusion-FP8` / `LFM3-1.7B-Diffusion-NVFP4`
- CUDA-graph capture: fixed-shape `B=32`, `small_block=8`

## Math

| Checkpoint | Mode | B | small_block | taus | TPF | tok/s | peak_VRAM | score | retention vs AR |
|----------|------|---|-------------|------|-----|-------|-----------|-------|-----------------|
| BF16 | spd | 32 | 8 | TODO | TODO | TODO | TODO | TODO | TODO |
| BF16 | ar_trust | 32 | 8 | TODO | TODO | TODO | TODO | TODO | 100% |
| FP8 | spd | 32 | 8 | TODO | TODO | TODO | TODO | TODO | TODO |
| NVFP4 | spd | 32 | 8 | TODO | TODO | TODO | TODO | TODO | TODO |

## Code

| Checkpoint | Mode | B | small_block | taus | TPF | tok/s | peak_VRAM | score | retention vs AR |
|----------|------|---|-------------|------|-----|-------|-----------|-------|-----------------|
| TODO rows as above | | | | | | | | | |

## IFEval

| Checkpoint | Mode | B | small_block | taus | TPF | tok/s | peak_VRAM | score | retention vs AR |
|----------|------|---|-------------|------|-----|-------|-----------|-------|-----------------|
| TODO rows as above | | | | | | | | | |

## BFCL / XML tool_call

| Checkpoint | Mode | B | small_block | taus | TPF | tok/s | peak_VRAM | valid_XML% | retention vs AR |
|----------|------|---|-------------|------|-----|-------|-----------|------------|-----------------|
| TODO rows as above | | | | | | | | | |

`ar_trust` must report 100% valid `tool_call` XML / JSON.

## Quantization deltas vs BF16 dLM

| Family | FP8 delta (pts) | NVFP4 delta (pts) | Within <1.0 pt? |
|--------|-----------------|--------------------|-----------------|
| Math | TODO | TODO | TODO |
| Code | TODO | TODO | TODO |
| IFEval | TODO | TODO | TODO |
| BFCL/XML | TODO | TODO | TODO |

## Sign-off checklist

- [ ] All three checkpoints load cleanly via `trust_remote_code=True`
- [ ] Quantization delta vs BF16 dLM < 1.0 pt on every family
- [ ] SPD `TPF >= 4.5` on held-out trajectories
- [ ] `ar_trust` output bit-identical to AR decoding (lossless)
- [ ] Retention vs `LiquidAI/LFM3-1.7B`: >= 90% `spd`, >= 95% `ar_trust`, per family
- [ ] tok/s >= 3.5x over BF16 AR on at least one quantized export
- [ ] MCS-NVFP4 weight checksums bit-identical to NVFP4 base (only the 72 `input_scale` scalars differ)
- [ ] `MODEL_CARD.md` TODO placeholders filled with final numbers
- [ ] Artifacts uploaded; `colab stop -s lfm3-g4` executed; operator `L4` session untouched
