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

## Decoding limits

`lfm3_dlm.sample.generate` keeps one column per row and a single shared
sliding-window ring, so batched decoding is supported only for models
without sliding-attention layers; `B > 1` raises on a sliding model
instead of silently serving a lagging row the wrong window. Run those one
sequence at a time until per-row KV lands.

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

GPU training is driven by `configs/g4_run.yaml` (the delivered schedule) and the
runbooks in `docs/`; it is out of scope for CPU-only environments.


## Running the converted model

The conversion run writes an HF directory (`/content/export` on the VM, copied
to `lfm3-artifacts/` locally). It loads like any LFM3 checkpoint:

```python
from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.tokenizer import load_tokenizer
from lfm3_dlm.sample import generate

model = Lfm3ForCausalLM.from_pretrained("lfm3-artifacts/export").cuda().eval()
model.config.attention_mode = "causal"          # the sampler's clean stream
tok = load_tokenizer()
prompt = tok("Explain why the sky is blue.", return_tensors="pt").input_ids.cuda()

# exact: verified against the model's own top-1 before each commit
out = generate(model, prompt, mode="ar_trust", max_new_tokens=64, isd_width=4)
print(tok.decode(out.ids[0, prompt.shape[1]:], skip_special_tokens=True))
```

`mode="spd"` trades parity for tokens-per-forward; `hard_commit` tunes the
tradeoff (`generate(..., hard_commit=0.99)`). `scripts/parity_sweep.py` measures
the real number against AR greedy for any checkpoint, which is the only way to
know which mode clears a parity bar.

## Measured (shipped checkpoint)

| mode | tokens/forward | tok/s | vs plain AR | parity |
|---|---|---|---|---|
| plain AR greedy | 1.00 | 32.2 | 1.00x | — |
| `ar_trust`, width 4 | 1.38 | **61.9** | **1.93x** | **1.000** |
| `spd`, block 8 / small 4 | 1.73 | 94.0 | 2.92x | 0.167 |
| `spd`, block 32 / small 32 | 9.14 | 195.3 | 6.06x | 0.042 |

`ar_trust` is the usable mode. The `spd` rows buy tokens per forward and lose
the text: at every commit rate tested the output collapses into repetition, so
treat them as throughput measurements rather than shippable settings. Quality
evidence (benchmarks, held-out loss, causal perplexity) and the full experiment
record are in `docs/runs/g4_phase0_phase1.md`.
