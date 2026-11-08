---
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-brainstorm
title: "Training efficiency for the LFM3 block-diffusion conversion"
date: 2026-11-08
execution: code
---

# Training efficiency for the LFM3 block-diffusion conversion

## Goal Capsule

Make the LFM3-1.7B → block-diffusion conversion fit its compute budget on a
single G4 by removing the measured waste in the training step and by adopting
multi-mask context sharing (MMT), gated behind a matched-compute pilot that has
to beat today's recipe before the 300M-token run is allowed to start.

## Product Contract

### Summary

The conversion pipeline is correct but 4.7x too slow and 40% too memory-hungry
to run as planned: 4.6k corpus tokens/s and 90 GB peak against a 22k tok/s and
<65 GB gate, which turns a ~50 CU job into a ~180 CU one. Two independent
causes were measured, and the published dLLM literature names a third lever we
have not taken. This work removes all three, then proves the training-semantics
change is worth its risk on a short pilot before any long run.

### Problem Frame

Measured on a G4 (RTX PRO 6000 Blackwell, 96 GB), 2026-11-08:

- **Throughput 4.6k corpus tok/s** (9.3k physical), gate 22k.
- **Peak VRAM 90.1 GB** at 4x4096, gate <65 GB.
- **FP8 MLPs gave nothing**: 9.09k vs 9.26k tok/s — the step is not
  matmul-bound, it runs at ~19% of BF16 peak.
- **Dense masks**: `create_dlm_masks_by_layer_type` materializes
  `[B,1,S,S]` additive masks (134 MB each at 4x4096) every forward, which also
  keeps SDPA off its flash kernel. `masks.dlm_mask_mod` exists for the
  FlexAttention path, is unit-tested for parity with `build_dlm_bool_mask`, and
  is never called.
- **No activation checkpointing**: every layer retains its intermediates;
  `FrozenMLP`'s `y13` alone is 503 MB per layer at 4x4096.
- **Cross-entropy falls back to materializing logits**: `cut_cross_entropy` is
  not installed on the run VM, so `sliced_lse` does `hidden.float() @
  classifier.float().t()` — ~3-6 GB of fp32 per step plus a 1 GB classifier
  cast. The comparable hybrid-BDLM release keeps a tiled linear CE as its
  default for exactly this reason (Chaturvedi et al., arXiv 2607.02805).
- **Compute per supervised token is low by construction**: each packed row
  carries one clean copy and two noisy copies, and MMT (Pan et al., EMNLP 2026)
  shows that block-diffusion training spends roughly half its FLOPs on tokens
  that never receive denoising supervision.

### Requirements

- **R1 — Throughput.** Sustained **>= 22k corpus tokens/s** on a G4 at
  `s_phys=4096`, measured with the same smoke the gate failed on.
- **R2 — Memory.** Peak **< 65 GB** at that setting, activation checkpointing
  allowed.
- **R3 — Attention path.** Block-diffusion masks run without materializing a
  dense `[B,1,S,S]` tensor; `dlm_mask_mod`/FlexAttention is the path, with the
  dense builder retained as a fallback for CPU and for correctness tests.
- **R4 — Loss path.** Cross-entropy never materializes `[positions, 125033]`
  as one fp32 block on the accelerator: either `cut_cross_entropy` is installed
  and used, or the compute falls back to position-chunked accumulation with the
  same numerics.
- **R5 — Compute per token.** The packed layout supports **K >= 2 mask patterns
  sharing one clean context in a single forward**, each mask isolated from its
  siblings and attending the shared clean context — the MMT construction — with
  K configurable and K=2 reproducing today's supervision.
- **R6 — Equivalence guarantees.** The causal AR path stays bit-identical to
  the vendored upstream behaviour, and the sampler's `build_step_mask` keeps
  agreeing with the training mask on the same stream/block layout. Neither may
  regress to buy throughput.

### Key Decisions

- **KD1 (user-directed) — Adopt compute-per-token sharing, not only speed
  fixes.** The gate is a compute problem, and MMT is the published answer to
  the specific waste this layout has. Rejected: fixing only the kernel-level
  overheads, which leaves the per-token utilisation where it is.
- **KD2 (user-directed) — Prove it on a matched-compute pilot before the long
  run.** Two 30M-token runs at equal CU: current packing vs MMT packing, then
  commit the winner to the 300M-token run. Rejected: adopting straight from the
  literature, which risks discovering the transfer failure at 300M tokens with
  no control run.
- **KD3 — 22k / <65 GB stay as requirements, not advisories.** They are the
  plan's own verification contract; weakening them in this document would move
  the failure downstream rather than fix it.

### Success Criteria

- **SC1** — The gate smoke reports >= 22k corpus tok/s and < 65 GB on a G4, with
  the command and settings recorded.
- **SC2** — Causal AR parity and sampler/training mask agreement tests still
  pass unchanged.
- **SC3** — The pilot produces a decision: MMT vs current packing, on equal CU,
  judged by held-out loss and the CAD trajectory, with the numbers recorded in
  `docs/runs/`. The 300M-token run starts only on an MMT win; a loss keeps the
  current recipe and the run proceeds with the speed fixes alone.

### Scope Boundaries

In scope: attention path, loss path, activation checkpointing, packed-layout
K-mask support, the pilot harness, and the run record.

Not in scope: changing the base model, changing the data mixture, the NVFP4/FP8
export path (U8), and anything that alters the causal AR guarantee. The
Post-hoc sampler work (batched sliding-window decoding) stays deferred.

### Dependencies

- A G4 Colab session (`colab new --gpu G4`) and its CU budget.
- `divelab/opdlm_train_data` + Nemotron subsets (verified reachable) for the
  pilot's corpus tokens.
- `cut_cross_entropy` availability on the VM (install or chunked fallback).

### Sources

- Pan, Zhao, Sun, Li, Wang, Han. *Fast Block Diffusion Training via Multi-Mask
  Context Sharing.* EMNLP 2026, pp. 46353–46370.
- Chaturvedi, Shroff, Suresh, Kang, Wen. *Training Hybrid Block Diffusion
  Language Models with Partial Bidirectionality.* arXiv 2607.02805 — tiled
  linear CE as the memory-safe default.
- Chen, Cai, Zhuang. *FlashBlock: Attention Caching for Efficient Long-Context
  Block Diffusion.* arXiv 2602.05305 — inference-side caching, not applicable
  to training.
- NVlabs/Fast-dLLM v2 (block-diffusion conversion reference implementation).
- This repo's measured gate results, `docs/runbooks/g4_conversion.md`.


## Planning Contract

### Key Technical Decisions

- **KTD1 — FlexAttention via `mask_mod`, dense builder retained.** `masks.dlm_mask_mod`
  already mirrors `build_dlm_bool_mask` and is unit-tested for parity. The
  block-diffusion path builds `create_block_mask(...)` per attention layer type
  and hands it to SDPA's flash path; the dense builder stays for CPU, tests, and
  any environment without FlexAttention. Governs R3.
- **KTD2 — Loss numerics unchanged, memory changed.** Cross-entropy keeps the
  same fp32 log-sum-exp over the sliced classifier; only the blocking changes
  (position chunks, or `cut_cross_entropy` when importable). Governs R4.
- **KTD3 — Checkpointing is a trainer flag, not a model change.** Decoder layers
  are wrapped at the call site so the vendored model stays byte-identical for
  the causal-parity guarantee. Governs R2, R6.
- **KTD4 — MMT is a collator/layout change with masks that isolate siblings.**
  `[P | R_clean | R_m1 … R_mK]`; each masked copy sees the shared clean prefix
  and its own block, never another copy. K=2 reproduces today's supervision
  exactly, so the pilot can A/B the layout with no other variable moving.
  Governs R5.
- **KTD5 — The gate is measured before the pilot, the pilot before the run.**
  No 300M-token spend until the smoke clears R1/R2, and none until the pilot
  picks a winner. Governs R1, SC1, SC3.

### High-Level Technical Design

```
collator ──► PackedSequence [P | R_clean | R_m1..R_mK]
                    │
                    ├─ doc_id / stream_id / block_id  ──► masks.py
                    │        dense builder  (CPU, tests)
                    │        dlm_mask_mod → create_block_mask (GPU flash)
                    │
trainer ── checkpointed decoder layers ──► hidden [B,S,H]
                    │
                    └─ loss_cad.sliced_lse  → chunked / CCE  ──► L_CAD + L_CE + L_AR
```

### Assumptions

- FlexAttention is available through the installed torch (2.14) on the G4; the
  dense path remains the fallback and the correctness reference.
- `cut_cross_entropy` may be absent; the chunked fallback must match its numerics
  closely enough for the parity tests to hold at `atol=1e-4`.

## Implementation Units

#### U1. FlexAttention block masks for the training forward

- **Goal.** Block-diffusion attention runs on the flash path with no dense
  `[B,1,S,S]` materialization on the accelerator.
- **Requirements.** R3, R6
- **Files.** `lfm3_dlm/masks.py`, `lfm3_dlm/modeling_lfm3.py`,
  `lfm3_dlm/configuration_lfm3.py`, `tests/test_masks_and_swa.py`
- **Approach.** Add `create_dlm_block_masks(config, position_ids, doc_ids,
  stream_ids, block_size, device)` returning `create_block_mask(dlm_mask_mod(...))`
  per attention layer type; in `create_dlm_masks_by_layer_type`, return those when
  `config.use_flex_attention` and torch exposes `create_block_mask`. Keep the
  dense builder as-is. No change to the causal path.
- **Test scenarios.** BlockMask materialized to dense equals
  `build_dlm_bool_mask` for a packed layout; a model with
  `use_flex_attention=True` produces logits matching the dense path within
  `atol=1e-3`; CPU falls back to dense.
- **Verification.** `pytest tests/test_masks_and_swa.py -q`; smoke throughput up.

#### U2. Chunked classifier loss

- **Goal.** No `[positions, 125033]` fp32 block is ever allocated on the GPU.
- **Requirements.** R4
- **Files.** `lfm3_dlm/train/loss_cad.py`, `tests/test_loss_cad.py`
- **Approach.** Keep CCE when importable; otherwise compute `lse` in position
  chunks (default 1024 rows) with a running log-sum-exp in fp32, and reuse a
  single cached fp32 classifier copy per step instead of `classifier.float()` per
  call.
- **Test scenarios.** Chunked `sliced_lse` equals the unchunked result at
  `atol=1e-5`; a large-M case runs without exceeding a set allocation budget
  (assert peak via `torch.cuda.max_memory_allocated` on the smoke, or a
  `torch.profiler`-free size assertion on CPU).
- **Verification.** `pytest tests/test_loss_cad.py -q`.

#### U3. Activation checkpointing at the trainer call site

- **Goal.** Peak VRAM drops below the 65 GB gate without touching vendored code.
- **Requirements.** R2, R6
- **Files.** `lfm3_dlm/train/trainer.py`, `configs/g4_300m.yaml`
- **Approach.** `activation_checkpointing: true` in config; wrap each decoder
  layer's forward in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
  while training. Model file untouched, so causal parity is unaffected.
- **Test scenarios.** Tiny model produces the same loss with and without the flag
  at `atol=1e-4`; smoke peak VRAM lower with the flag on.
- **Verification.** `pytest tests/test_collator_packing.py -q`; smoke VRAM.

#### U4. MMT packing: K masks sharing one clean context

- **Goal.** A packed row carries K mask patterns over one clean context, each
  supervised and isolated from its siblings.
- **Requirements.** R5, R6
- **Files.** `lfm3_dlm/train/collator.py`, `lfm3_dlm/masks.py`,
  `tests/test_collator_packing.py`, `tests/test_masks_and_swa.py`
- **Approach.** Generalize the collator's two noisy streams to K, each with its
  own Bernoulli draw; block ids per stream; masks keep `same_stream` isolation
  and add cross-mask exclusion (a masked copy never sees another masked copy's
  block, only the clean prefix and its own). K=2 must reproduce today's tensors
  bit-for-bit.
- **Test scenarios.** K=2 equals the current layout exactly; K=4 yields four
  disjoint noisy blocks each attending the clean prefix; supervision count per
  response token is unchanged as K grows (each token supervised once).
- **Verification.** `pytest tests/test_collator_packing.py tests/test_masks_and_swa.py -q`.

#### U5. Gate smoke and the 30M-token A/B pilot

- **Goal.** A recorded measurement that decides K and the layout before the
  long run.
- **Requirements.** R1, SC1, SC3
- **Files.** `scripts/bench_training_step.py`, `scripts/pilot_ab.py`,
  `docs/runbooks/g4_conversion.md`, `docs/runs/g4_pilot.md`
- **Approach.** `bench_training_step.py` reports corpus tok/s and peak VRAM at
  `s_phys=4096` under the current flags. `pilot_ab.py` runs two 30M-token pilots
  at equal CU — current packing vs MMT packing — with fixed seeds, writing loss
  and CAD traces plus a held-out loss. Decision rule: MMT wins only on a lower
  held-out loss at equal CU with no throughput regression.
- **Test scenarios.** The bench script prints both numbers and exits non-zero
  below the gate (so CI-style use is possible); the pilot writes a machine-
  readable summary.
- **Verification.** `docs/runs/g4_pilot.md` contains both runs, the decision, and
  the CU spent.

## Verification Contract

- `TMPDIR=<disk> python -m pytest tests/ -q` must stay green, 91+ tests.
- Causal AR parity (`tests/test_masks_and_swa.py`, `tests/test_dual_sampler.py`)
  must pass unchanged — no test in those files may be deleted or weakened.
- Gate smoke: `python scripts/bench_training_step.py --s-phys 4096` reports
  `>= 22k` corpus tok/s and `< 65 GB` peak on a G4.
- Pilot: `docs/runs/g4_pilot.md` records both arms, the metric table, and the
  winner.

## Definition of Done

- R1–R6 met, with the gate smoke and pilot numbers recorded in `docs/runs/`.
- Causal AR parity and sampler/training mask agreement still pass unchanged.
- Dead-end code from abandoned attempts removed; no flag defaults to an
  unverified path (FP8 stays opt-in until it earns its place).
- The 300M-token run is launched only after the pilot's decision rule fires.
