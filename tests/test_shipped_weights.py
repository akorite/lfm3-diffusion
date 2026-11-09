"""The pulled `final-weights.pt` must be the model it claims to be.

Only its md5 had been checked: a file can hash consistently and still be the
wrong tensor dict, or carry a mask row that was silently re-seeded from the pad
row. This reads it with `mmap=True` so 5.7 GB never lands in RAM, checks the
key set, and compares the two embedding rows that matter.

Skips when the artifact is absent, so a clean checkout stays green.
"""
import os
from pathlib import Path

import pytest
import torch

MASK_ROW, PAD_ROW = 125_033, 124_893
WEIGHTS = Path(os.environ.get("LFM3_WEIGHTS",
                              "/home/ubuntu/lfm3-artifacts/final-weights.pt"))
EMBED = "model.embed_tokens.weight"


def _load():
    if not WEIGHTS.exists():
        pytest.skip(f"{WEIGHTS} not present")
    # mmap keeps the 5.7 GB lazier than RAM: only touched pages load
    return torch.load(WEIGHTS, mmap=True)


def test_weights_has_the_expected_key_set():
    sd = _load()
    assert len(sd) == 311, f"expected 311 tensors, found {len(sd)}"
    assert EMBED in sd
    assert int(sd[EMBED].shape[0]) > MASK_ROW


def test_weights_mask_row_is_not_the_pad_row():
    sd = _load()
    emb = sd[EMBED]
    mask_row = emb[MASK_ROW]
    pad_row = emb[PAD_ROW]
    assert not torch.allclose(mask_row, pad_row), (
        "the mask embedding row equals the pad row: the diffusion row was lost"
    )
