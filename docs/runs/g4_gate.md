# G4 gate measurements — 2026-11-08

Hardware: `colab new -s lfm3-g4 --gpu G4` → RTX PRO 6000 Blackwell, 97,887 MiB,
CC 12.0, torch 2.14.1+cu130, transformers 5.21.0. Batch 1–4 packed rows of
`s_phys=4096`, frozen MLPs, BF16 unless stated.

| Config | s/step | physical tok/s | corpus tok/s | peak VRAM |
|---|---|---|---|---|
| dense masks, mb=1 (x2 rows) | 0.837 | 9,786 | 4,893 | 31.8 GB |
| dense masks, mb=4 | 1.770 | 9,258 | 4,629 | **90.1 GB** |
| FP8 MLPs, mb=4 | 1.802 | 9,090 | 4,545 | 90.1 GB |
| FlexAttention, mb=1 | 0.757 | 10,821 | 5,410 | 29.1 GB |
| **FlexAttention + CCE, mb=1** | **0.638** | **12,840** | **6,420** | **24.5 GB** |
| FlexAttention + CCE, mb=2 (4 rows) | 1.205 | 13,596 | 6,798 | 48.4 GB |
| FlexAttention + CCE, mb=4 (8 rows) | 2.535 | 12,928 | 6,464 | 76.7 GB |
| dense + native gradient checkpointing, mb=1 | 0.925 | 8,856 | 4,428 | 20.5 GB |
| FlexAttention + layer checkpointing | — | fails | — | — |

`RuntimeError: block_mask was created for a smaller length than you're using it
for` — FlexAttention and `torch.utils.checkpoint` are incompatible in this torch
build; the two levers cannot be combined as written.

## What this says

- **The mask was not the bottleneck.** FlexAttention buys ~10%, and the
  traceback shows it falling back to `sdpa_dense` / `math_attention` rather than
  the flash kernel, so the dense-mask cost is not what dominates the step.
- **Achieved compute:** the 4-row step does ≈168 TFLOP in 1.77 s ≈ 95 TFLOPS
  (~19% of BF16 peak); the 1-row flex step does ≈42 TFLOP in 0.757 s ≈ 55 TFLOPS
  (~11%). Flex is ~10% faster per token; single-row batching costs MFU, so the
  two effects partly cancel. FP8 giving 0% confirms it is not matmul-bound.
- **VRAM is fixable, throughput is not, with the levers tried.** mb=1 lands at
  29 GB, well under the 65 GB gate; the throughput gate stays at ~5.4k against 22k.
- **`cut_cross_entropy` is worth installing** (it was absent from the VM): +19%
  throughput and −4.6 GB over the chunked fallback. It is now on the VM and is
  the plan's intended path.
- **Microbatching does not help**: 6.4–6.8k corpus tok/s at mb=1/2/4 while VRAM
  climbs 33.8 → 48.4 → 76.7 GB. mb=1 (or 2) is the operating point.
- **Native gradient checkpointing works but costs ~30%** (4.4k vs 6.4k) for
  −4 GB, and VRAM is already under the gate, so it stays off.
- **Implication for the plan's budget:** 300M corpus tokens at 6.5k tok/s is
  ~12.8 h ≈ 129 CU, against the plan's 46–55 CU. It fits the 224 CU balance but
  not the phase budget.

## Not yet tried

- Larger effective batch with activation checkpointing *without* FlexAttention
  (checkpointing works on the dense path; the combination that failed was
  flex+checkpointing).
- `torch.compile` on the step.
- Profile (`torch.profiler`) to name the dominant kernel instead of inferring
  it from scaling.
