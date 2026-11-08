"""A Medusa/EAGLE-style multi-token draft head for AR-Trust.

AR-Trust verifies every emitted token, so a draft only has to *propose*: a wrong
proposal costs one rejected verify step, never a wrong token. That is why draft
heads buy speed at zero quality loss (Medusa 2.3x, EAGLE 2.8x) - and why the
proposer, not the verifier, is the honest place to spend effort here.

The head reads the model's hidden state at the last committed position and
proposes the next `k` tokens. The model's own output head is reused (frozen), so
the trainable part is a residual MLP plus one small block over `k` slots rather
than `k` full vocab matrices.
"""
from __future__ import annotations

import torch
from torch import nn


class DraftHead(nn.Module):
    """`hidden_at_committed` -> logits for the next `k` positions."""

    def __init__(self, hidden_size: int, k: int = 4, inner: int = 0):
        super().__init__()
        self.k = k
        inner = inner or hidden_size
        self.pre = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_size),
        )
        self.slot = nn.Parameter(torch.zeros(k, hidden_size))
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=inner,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, lm_head) -> torch.Tensor:
        """hidden `[B, H]` (or `[H]`) -> logits `[B, k, V]`."""
        squeeze = hidden.dim() == 1
        if squeeze:
            hidden = hidden.unsqueeze(0)
        x = hidden.to(self.slot.dtype)
        h = x + self.pre(x)
        seq = h.unsqueeze(1) + self.slot.unsqueeze(0)          # [B, k, H]
        seq = self.block(seq)
        seq = self.norm(seq)
        logits = lm_head(seq.to(lm_head.weight.dtype))          # shared frozen head
        return logits[0] if squeeze else logits
