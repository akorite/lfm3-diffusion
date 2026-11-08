# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared-prefix block-diffusion masks for the hybrid LFM3 stack.

The packed training layout concatenates per-document streams
``[P | R_clean | R_m1 | R_m2]`` in one row: ``position_ids`` carry the
*logical* position (restart per stream, restart per document),
``stream_ids`` mark stream 0 (prompt + clean response) versus the noisy
streams 1 and 2, and ``doc_ids`` isolate packed documents.

`block_diffusion` allowance:

- stream-0 queries: causal within their document;
- noisy queries: bidirectional inside their own ``block_size`` block and
  causal to stream-0 blocks strictly earlier than their block;
- everything else (other streams, other documents, later blocks) is cut.

Sliding-window layers gate on the *logical* position difference
``0 <= lp_q - lp_kv < sliding_window`` (same-block noisy pairs exempt so
the block stays bidirectional); gating on the physical tensor index
would blind noisy blocks to clean context because the packed streams
offset them by the clean-stream length.
"""
from __future__ import annotations

import torch

DEFAULT_BLOCK_SIZE = 32
# Warmup anneal schedule for the diffusion block size (continued anneal
# picks up where the schedule ends); the mask math only needs the size.
BLOCK_WARMUP_SCHEDULE = (4, 8, 16, 32)


def pack_stream_metadata(
    position_ids: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
    stream_ids: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """Normalize packing metadata: derive documents from restarts, default
    a single clean stream, and compute the per-stream logical block index."""
    if doc_ids is None:
        restarts = torch.diff(position_ids, prepend=position_ids[:, :1] - 1, dim=-1) != 1
        doc_ids = restarts.to(torch.long).cumsum(-1)
    if stream_ids is None:
        stream_ids = torch.zeros_like(position_ids)
    block_ids = torch.div(position_ids, block_size, rounding_mode="floor")
    return doc_ids.long(), stream_ids.long(), block_ids


def build_dlm_bool_mask(
    position_ids: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
    stream_ids: torch.Tensor | None = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    mode: str = "block_diffusion",
    sliding_window: int | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense boolean ``[batch, 1, seq, seq]`` allow-mask (CPU fallback for
    FlexAttention's `mask_mod`; see `dlm_mask_mod` for the indexed form)."""
    doc_ids, stream_ids, block_ids = pack_stream_metadata(position_ids, doc_ids, stream_ids, block_size)

    lp_q = position_ids[:, :, None]
    lp_kv = position_ids[:, None, :]
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    q_clean = (stream_ids == 0)[:, :, None]
    k_clean = (stream_ids == 0)[:, None, :]

    if mode == "causal":
        allowed = same_doc & (lp_kv <= lp_q)
        if sliding_window is not None:
            allowed &= (lp_q - lp_kv) < sliding_window
    elif mode == "block_diffusion":
        same_stream = stream_ids[:, :, None] == stream_ids[:, None, :]
        block_q = block_ids[:, :, None]
        block_kv = block_ids[:, None, :]
        own_noisy_block = same_stream & ~q_clean & ~k_clean & (block_q == block_kv)
        clean_causal = q_clean & k_clean & (lp_kv <= lp_q)
        earlier_clean = ~q_clean & k_clean & (block_kv < block_q)
        allowed = same_doc & (clean_causal | own_noisy_block | earlier_clean)
        if sliding_window is not None:
            allowed &= own_noisy_block | ((lp_q - lp_kv) < sliding_window)
    else:
        raise ValueError(f"mode must be 'causal' or 'block_diffusion'; got {mode!r}")

    if padding_mask is not None:
        allowed &= padding_mask[:, None, :].bool()
    return allowed[:, None, :, :]


def dlm_mask_mod(
    position_ids: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
    stream_ids: torch.Tensor | None = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    mode: str = "block_diffusion",
    sliding_window: int | None = None,
):
    """FlexAttention `mask_mod`-style closure over the packing metadata:
    ``mask_mod(b, h, q, kv) -> bool`` on logical positions, never physical
    indices."""
    doc_ids, stream_ids, block_ids = pack_stream_metadata(position_ids, doc_ids, stream_ids, block_size)

    def mask_mod(b, h, q, kv):
        same_doc = doc_ids[b, q] == doc_ids[b, kv]
        clean_q = stream_ids[b, q] == 0
        clean_kv = stream_ids[b, kv] == 0
        if mode == "causal":
            allowed = same_doc & (position_ids[b, kv] <= position_ids[b, q])
        else:
            own_block = (
                (stream_ids[b, q] == stream_ids[b, kv])
                & ~clean_q
                & ~clean_kv
                & (block_ids[b, q] == block_ids[b, kv])
            )
            allowed = same_doc & (
                (clean_q & clean_kv & (position_ids[b, kv] <= position_ids[b, q]))
                | own_block
                | (~clean_q & clean_kv & (block_ids[b, kv] < block_ids[b, q]))
            )
            if sliding_window is not None:
                allowed = allowed & (own_block | ((position_ids[b, q] - position_ids[b, kv]) < sliding_window))
        if mode == "causal" and sliding_window is not None:
            allowed = allowed & ((position_ids[b, q] - position_ids[b, kv]) < sliding_window)
        return allowed

    return mask_mod


def additive_mask(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean allow-mask to the additive float mask eager/SDPA consume."""
    return torch.zeros_like(bool_mask, dtype=dtype).masked_fill(~bool_mask, torch.finfo(dtype).min)


def create_dlm_block_masks(
    config,
    position_ids: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
    stream_ids: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    device=None,
) -> dict:
    """Per-layer-type `BlockMask`s for FlexAttention (no dense S x S tensor).

    Same rule as `build_dlm_bool_mask`, expressed as a `mask_mod` so the
    kernel skips blocked quadrants instead of materializing them.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch, seq = position_ids.shape
    mapping: dict = {}
    for layer_type in ("full_attention", "sliding_attention"):
        if layer_type not in config.layer_types:
            continue
        mod = dlm_mask_mod(
            position_ids, doc_ids, stream_ids,
            block_size=block_size, mode=config.attention_mode,
            sliding_window=config.sliding_window if layer_type == "sliding_attention" else None,
        )
        mapping[layer_type] = create_block_mask(
            mod, batch, None, seq, seq, device=device or position_ids.device,
        )
    if "conv" in config.layer_types:
        mapping["conv"] = None
    return mapping


def create_dlm_masks_by_layer_type(
    config,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    doc_ids: torch.Tensor | None = None,
    stream_ids: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    dtype: torch.dtype | None = None,
) -> dict:
    """Per-layer-type mask dict mirroring `create_masks_by_layer_type`.

    The `"conv"` entry is always present (the layer loop indexes it) and
    carries the plain 2D padding mask; document isolation for the short
    convs comes from `doc_ids` (`seq_idx`), not from this mask.
    """
    if position_ids is None:
        raise ValueError("block-diffusion masks need `position_ids` (logical positions per stream)")
    if dtype is None:
        dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.get_default_dtype()

    padding = attention_mask if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 else None
    if getattr(config, "use_flex_attention", False):
        try:
            mapping = create_dlm_block_masks(
                config, position_ids, doc_ids, stream_ids,
                block_size=block_size, device=position_ids.device,
            )
            mapping["conv"] = padding
            return mapping
        except Exception:  # no FlexAttention in this build: dense is the reference
            pass
    mapping = {}
    layer_types = config.layer_types
    if "full_attention" in layer_types:
        mapping["full_attention"] = additive_mask(
            build_dlm_bool_mask(
                position_ids, doc_ids, stream_ids,
                block_size=block_size, mode=config.attention_mode, padding_mask=padding,
            ),
            dtype,
        )
    if "sliding_attention" in layer_types:
        mapping["sliding_attention"] = additive_mask(
            build_dlm_bool_mask(
                position_ids, doc_ids, stream_ids,
                block_size=block_size, mode=config.attention_mode,
                sliding_window=config.sliding_window, padding_mask=padding,
            ),
            dtype,
        )
    if "conv" in layer_types:
        mapping["conv"] = padding
    return mapping


__all__ = [
    "BLOCK_WARMUP_SCHEDULE",
    "DEFAULT_BLOCK_SIZE",
    "additive_mask",
    "build_dlm_bool_mask",
    "create_dlm_masks_by_layer_type",
    "dlm_mask_mod",
    "pack_stream_metadata",
]
