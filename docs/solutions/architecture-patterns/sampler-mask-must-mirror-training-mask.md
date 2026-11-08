---
title: Sampler masks must mirror the training mask, not just look reasonable
date: 2026-11-08
category: architecture-patterns
module: block-diffusion sampler / training masks
problem_type: architecture_pattern
component: inference_sampler
severity: high
applies_when: a serving-time decode loop builds its own attention masks instead of calling the training mask builder
tags: [block-diffusion, attention-mask, cache-index, train-serve-skew]
---

# Sampler masks must mirror the training mask, not just look reasonable

## Context

Block-diffusion training (`lfm3_dlm/masks.py`) packs `[P | R_clean | R_m1 |
R_m2]` per document and lets a noisy query see clean context only from
*earlier* blocks, plus its own block's noisy slots bidirectionally. The
cached sampler (`lfm3_dlm/sample/dual_sampler.py`) cannot call that builder
directly — it has a KV cache, per-row commit fronts and fed-slot positions —
so it re-implements the rule in `build_step_mask`. Three separate mistakes
survived a full code review because each looked locally reasonable:

1. **Block ids on a different grid.** Training computes blocks as
   `logical_position // block_size`; the sampler used
   `(position - prompt_len) // block_size`. They agree only when the prompt
   length happens to be a multiple of the block size, so real prompts
   silently trained and served different co-visibility sets.
2. **Clean same-block keys left visible to noisy queries.** The causal term
   `fed & (kv_pos <= q_pos)` admitted a clean token in the noisy block — the
   very token the block was predicting. Training's mask excludes it.
3. **Cache index treated as a logical position.** With per-row commit fronts
   a lagging row's K/V lands at physical indices past its own logical
   position, so labelling cached slots `arange(appended)` handed every row
   the fastest row's positions.

## Guidance

- When a serving loop re-derives a mask the trainer owns, write the rule
  once in prose and assert it in both places. A cheap direct assertion on
  `build_step_mask` (noisy query vs same-block clean key must be `-inf`;
  same-block noisy key must be `0`) catches drift that end-to-end logit
  comparisons hide, because those build both sides with the same function.
- Block ids are logical positions divided by the block size, on both sides.
  A stream-relative grid is a bug that only shows up for prompts whose
  length is not a multiple of the block size — pick a test prompt that is
  deliberately misaligned.
- A cache slot is not a position. Keep a per-row `cache_pos` (logical
  position per physical slot, `-1` where a slot was invalidated) and mask on
  it; crop by the last slot any row still needs, never by
  `max(committed)`, because that count is logical.
- Re-feeding a position must invalidate the copies written earlier (rejected
  drafts), or the stale entry returns as committed context once the row
  commits past it.

## Applicability

Applies to any cached decode loop whose masks are derived from training-time
packing metadata — speculative decoding, block/parallel diffusion, and
prefix-sharing batched serving all have the same three traps. It does not
apply when the sampler can call the trainer's mask builder unchanged.

## Prevention

`tests/test_dual_sampler.py` now pins the mask rule directly
(`test_step_mask_restricts_noisy_queries_to_earlier_clean_blocks`), the
logical block grid (`test_noisy_feeds_use_logical_block_ids`), the window
alignment (`test_ar_trust_parity_past_the_sliding_window`) and batched
row independence across several seeds
(`test_batched_rows_match_their_single_row_runs`). Batched decoding with
sliding-window layers is refused rather than silently served from one
shared ring.
