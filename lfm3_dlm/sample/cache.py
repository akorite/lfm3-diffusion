"""Prefix KV + Bx cache for the dual-mode sampler.

Wraps `DynamicCache` with committed-length bookkeeping:

- Full-attention layers keep the whole committed prefix.
- Sliding-attention layers are restricted to the trailing window via
  `DynamicSlidingWindowLayer` (a `sliding_window` token ring).
- ShortConv layers keep the last `conv_kernel_size` `B*x` inputs; past
  recording is activated so a rejection can roll the cache back.

Draft/noisy positions appended during a step are rolled back with
`crop`: committed K/V and Bx are truncated to the accepted length, so
rejected drafts never enter the clean prefix.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


class PrefixCache:
    """Dynamic cache + per-row committed lengths for block diffusion / ISD."""

    def __init__(self, config):
        self.cache = DynamicCache(config=config)
        # past recording keeps full conv/SWA history until `crop`, which is
        # what makes rollback to the accepted length possible at all
        self.cache.activate_past_recording()
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window
        self.conv_kernel_size = config.conv_L_cache
        # total positions ever appended (== cache sequence length)
        self.appended = 0

    # -- maintenance ---------------------------------------------------------

    def crop(self, new_appended: int) -> None:
        """Drop the last `appended - new_appended` positions from every layer.

        K/V rows and the recorded `B*x` history are truncated; ShortConv and
        SWA layers additionally shrink back to their working window.
        """
        n = self.appended - new_appended
        if n < 0:
            raise ValueError("cannot crop forward")
        if n:
            self.cache.crop(-n)
            self.appended = new_appended
        # `crop(0)` is not a no-op for windowed layers: it restores the
        # conv state / SWA ring to their working size after recording.
        self.cache.crop(0)

    def reset(self) -> None:
        self.cache.reset()
        self.appended = 0
