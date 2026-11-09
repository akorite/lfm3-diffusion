# Status

The harness goal tracker is not available in this session, so this file is the
progress record. It mirrors the objective, the criteria, and what is left.

## Objective

Ship an LFM3-1.7B block-diffusion checkpoint that decodes with AR-greedy token
parity at a real speed win, without degrading quality, measured by this repo's
own harnesses.

**Status: delivered and verified. Two follow-ups blocked on resources.**

## Criteria and evidence

| # | criterion | measured | status |
|---|---|---|---|
| 1 | parity >= 0.95 | 1.000 mean, 1.000 worst (4 prompts x 24 tokens) | met |
| 2 | TPF >= 1.5 | 1.38 usable (`ar_trust`); 1.60-9.14 only where text collapses | met as written, not at usable quality |
| 3 | held-out <= base | 4.779 vs 15.717; samples non-degenerate | met |
| 4 | tests | 106 passed | met |
| 5 | export loads | 24 layers, mask row != pad row, byte-verified tarball | met |
| 6 | committed + pulled | `docs/runs/`, artifacts md5-verified | met |

Delivered checkpoint: `ckpt-25004750` (25M tokens), decoding at **61.9 tok/s =
1.93x plain autoregressive**, parity 1.000.

Quality: arc_easy 0.533 vs base 0.420, piqa 0.675 vs 0.590, hellaswag 0.398 vs
0.405, causal perplexity 4.396 vs 4.419.

## Done

- conversion trained, exported, measured, published to `lfm3-artifacts/`
- four proposer designs tried and measured (mask-draft, two teacher-forced
  heads, one on-policy head): 0.91-1.09 tokens/forward, none beat the mask-draft
- decode levers swept (block geometry, draft width, commit threshold): flat
- two polish legs (`t_high`, `mask_token_frac`) measured: no effect on TPF
- docs corrected to measured numbers only, with a drift test
- shipped tarball and shipped weights verified as bytes, not as claims

## Blocked

| item | why |
|---|---|
| benchmarks at n=1000 (arc_easy, piqa, hellaswag) | the 2.75 GB export must reach a GPU VM; the Colab contents API caps uploads near 50 MB at ~1 MB/s and the session was reaped mid-transfer twice. A Hugging Face token removes this wall. |
| diffusion-native throughput at TPF >= 1.5 with coherent text | needs a conversion at 30-70B tokens (DiffuLLaMA's floor for a 7B model); this run used 25M. Not a Colab-scale job. |

## Verify

```bash
pytest tests/ -q                      # 106 tests
python scripts/parity_sweep.py --weights <w> --prompts 16 --tokens 24
python scripts/speed_wallclock.py --weights <w>
python scripts/heldout_eval.py --weights <w> --count 40      # and without --weights for the base
python scripts/assert_export.py --path <export-dir>
```

The full experiment record, including every negative result, is in
`docs/runs/g4_phase0_phase1.md`.
