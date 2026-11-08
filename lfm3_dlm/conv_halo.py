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
"""Clean-context halo for the width-7 depthwise ShortConv.

Packed `[P | R_clean | R_m1 | R_m2]` rows place the noisy streams after
the clean stream, so the causal conv window at a noisy position would
physically read the tail of the clean stream and, for blocks past the
first, *other noisy blocks*. Instead each noisy block must read the
``kernel_size - 1`` clean tokens that precede it at logical positions
``lp - 6 .. lp - 1`` in stream 0 of the same document (zeroed across
document boundaries — those logical positions do not exist).

`build_conv_halo` emits a per-position gather index for the whole
causal window (`[batch, seq, kernel_size]`, last entry the position
itself) plus a validity mask; `halo_conv1d` gathers the window and
contracts it with the depthwise weights, replacing `Conv1d` where the
halo is given. `in_proj`/`out_proj` still run once on the un-haloed
tokens.
"""
from __future__ import annotations

import torch


def build_conv_halo(
    position_ids: torch.Tensor,
    doc_ids: torch.Tensor,
    stream_ids: torch.Tensor,
    kernel_size: int = 7,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position gather index `(batch, seq, kernel_size)` into the row,
    and a boolean validity mask (invalid halo entries gather position 0
    and must be zeroed before the depthwise product)."""
    batch, seq = position_ids.shape
    device = position_ids.device
    clean = stream_ids == 0

    # `slot` keys each token by (document, logical position) so noisy
    # blocks can resolve their clean context; non-clean columns land in a
    # dump slot that is never read back.
    max_pos = int(position_ids.max().item()) if position_ids.numel() else 0
    n_slots = (int(doc_ids.max().item()) + 1) * (max_pos + 1) + 1 if doc_ids.numel() else 1
    slot = doc_ids * (max_pos + 1) + position_ids
    dump = torch.full_like(position_ids, n_slots - 1)
    src = torch.full((batch, n_slots), -1, dtype=torch.long, device=device)
    cols = torch.arange(seq, device=device)[None, :].expand(batch, seq)
    src.scatter_(1, torch.where(clean, slot, dump), cols)

    halo = kernel_size - 1
    offsets = torch.arange(halo, 0, -1, device=device)  # kernel_size-1 .. 1 back from the position
    wanted = position_ids[:, :, None] - offsets[None, None, :]  # logical positions lp - halo .. lp - 1
    prev_idx = torch.arange(seq, device=device)[None, :, None] - offsets[None, None, :]

    in_doc = wanted >= 0
    resolved = src.gather(1, (doc_ids[:, :, None] * (max_pos + 1) + wanted.clamp(min=0)).reshape(batch, -1))
    resolved = resolved.reshape(batch, seq, halo)

    clean_idx = prev_idx.clamp(min=0).expand(batch, seq, halo)
    prev_doc = doc_ids.gather(1, clean_idx.reshape(batch, -1)).reshape(batch, seq, halo)
    clean_valid = (prev_idx >= 0) & (doc_ids[:, :, None] == prev_doc)

    # Noisy window columns resolve the same `(doc, lp)` slots, but only the
    # token's own block columns and the clean halo of block `block_q - 1`
    # may be read: columns of other noisy blocks are invalid.
    block_q = torch.div(position_ids, block_size, rounding_mode="floor")
    wanted_block = torch.div(wanted.clamp(min=0), block_size, rounding_mode="floor")
    # `resolved < 0` already cuts cross-document reads; among the in-document
    # columns only the token's own block and the clean halo of `block_q - 1` may be read.
    in_block = wanted_block == block_q[:, :, None]
    halo_ok = in_doc & (wanted_block == block_q[:, :, None] - 1)
    prev_stream = stream_ids.gather(1, clean_idx.reshape(batch, -1)).reshape(batch, seq, halo)
    noisy_valid = (
        (in_block & (wanted >= 0) & (prev_stream == stream_ids[:, :, None]))
        | (halo_ok & (resolved >= 0))
    )
    # own-block columns are the physical left neighbors (packed or sequential
    # layout alike); only the cross-block clean halo resolves through `src`
    noisy_idx = torch.where(in_block & (wanted >= 0), prev_idx.clamp(min=0), resolved.clamp(min=0))

    is_clean = clean[:, :, None]
    idx = torch.where(is_clean, clean_idx, noisy_idx)
    valid = torch.where(is_clean, clean_valid, noisy_valid)
    idx = torch.cat([idx, torch.arange(seq, device=device)[None, :, None].expand(batch, -1, -1)], dim=-1)
    valid = torch.cat([valid, valid.new_ones(batch, seq, 1)], dim=-1)
    return idx, valid


def halo_conv1d(
    hidden_states: torch.Tensor,
    halo_idx: torch.Tensor,
    halo_valid: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise width-`kernel` conv reading `(batch, hidden, seq)` through
    the `(batch, seq, kernel)` halo gather index — one einsum, no padding."""
    seq, kernel = halo_idx.shape[1], halo_idx.shape[2]
    windows = hidden_states[..., None].expand(-1, -1, seq, kernel).gather(
        2, halo_idx[:, None, :, :].expand(-1, hidden_states.shape[1], seq, kernel)
    )
    windows = windows * halo_valid[:, None].to(windows.dtype)
    out = torch.einsum("bdtk,dk->bdt", windows.to(weight.dtype), weight.squeeze(1))
    if bias is not None:
        out = out + bias[None, :, None]
    return out.to(hidden_states.dtype)


__all__ = ["build_conv_halo", "halo_conv1d"]
