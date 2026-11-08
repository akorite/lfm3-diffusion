# G4 gate measurements — 2026-11-08

Hardware: `colab new -s lfm3-g4 --gpu G4` → RTX PRO 6000 Blackwell, 97,887 MiB,
CC 12.0, torch 2.14.1+cu130, transformers 5.21.0. Batch 1–4 packed rows of
`s_phys=4096`, frozen MLPs, BF16 unless stated.

| Config | s/step | physical tok/s | corpus tok/s | peak VRAM |
|---|---|---|---|---|
| dense masks, mb=1 (x2 rows) | 0.837 | 9,786 | 4,893 | 31.8 GB |
| dense masks, mb=4 | 1.770 | 9,258 | 4,629 | **90.1 GB** |
| FP8 MLPs, mb=4 | 1.802 | 9,090 | 4,545 | 90.1 GB |
| **FlexAttention, mb=1** | **0.757** | **10,821** | **5,410** | **29.1 GB** |
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
- **Implication for the plan's budget:** 300M corpus tokens at 5.4k tok/s is
  ~15.4 h ≈ 155 CU, against the plan's 46–55 CU. The run fits the 234 CU balance
  but not the phase budget.

## Not yet tried

- Larger effective batch with activation checkpointing *without* FlexAttention
  (checkpointing works on the dense path; the combination that failed was
  flex+checkpointing).
- `torch.compile` on the step.
- Profile (`torch.profiler`) to name the dominant kernel instead of inferring
  it from scaling.
