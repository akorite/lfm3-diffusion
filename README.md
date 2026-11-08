# LFM3 Block-Diffusion

Converts `LiquidAI/LFM3-1.7B` into a dual-mode block-diffusion language model
(`B = 32` blocks): the post-trained checkpoint learns parallel denoising over
noisy token blocks while keeping its causal stream intact, so one checkpoint
supports both soft parallel decoding and lossless causal `AR-Trust`
verification (used automatically inside `<tool_call>` spans).

## Layout

- `lfm3_dlm/` — the package
  - `configuration_lfm3.py` / `modeling_lfm3.py` / `lfm3_conv.py` — the LFM3
    hybrid model (13 gated `ShortConv`, 5 `SWA-256`, 6 full-attention layers)
    extended with `attention_mode = "causal" | "block_diffusion"`. The mask
    token `<|mask|>` is bound to embedding row `125033`; the output classifier
    is sliced to rows `< 125033`.
  - `masks.py` — block-diffusion attention masks per layer type.
  - `conv_halo.py` — clean 6-token halo so `ShortConv` stays causal across
    packed document boundaries.
  - `tokenizer.py` — tokenizer loader that registers `<|mask|>` at id 125033.
  - `convert.py` — deterministic checkpoint converter: exports a
    self-contained HF directory (code + config + tokenizer + LICENSE) and
    merges trained mixer weights back into the unfused `w1/w2/w3` layout.
  - `train/` — shared-prefix collator, confidence-adaptive KL loss, static-FP8
    frozen-MLP wrapper, trainer loop.
  - `data/` — Top-16 teacher-logprob cache codec.
- `tests/` — CPU test suite (no CUDA required).
- `configs/` — GPU training configs (`g4_300m.yaml`); runbooks live under `docs/`.

## Tests

```bash
PYTHONPATH=<path-to-torch-env> python3 -m pytest tests/ -q \
    --basetemp=<dir-on-disk>   # only if /tmp is a small tmpfs
```

## Exporting a checkpoint
```bash
python3 -m lfm3_dlm.convert export <source-dir> <export-dir>
python3 -m lfm3_dlm.convert merge <base-dir> <trained-dir> <export-dir>
```

`export` produces a directory that loads standalone via
`AutoModelForCausalLM.from_pretrained(<export-dir>, trust_remote_code=True)`.
`merge` takes mixer/norm weights from a trained snapshot and re-applies them
onto the base checkpoint's frozen MLP/embedding weights.

GPU training is driven by `configs/g4_300m.yaml` and the runbooks in `docs/`;
it is out of scope for CPU-only environments.
