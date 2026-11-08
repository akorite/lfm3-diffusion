"""Tokenizer for `LiquidAI/LFM3-1.7B` with the diffusion mask token bound.

Row 125033 of the (tied) embedding is the first unused row after the last
added token (125032), so registering `<|mask|>` as a special token lands it
exactly there without touching `vocab_size` (kept 256-aligned at 128000).
"""
from pathlib import Path

from transformers import AutoTokenizer

MASK_TOKEN = "<|mask|>"
MASK_TOKEN_ID = 125_033

_ASSETS = Path(__file__).parent / "assets" / "lfm3_tokenizer"


def load_tokenizer(path: str | Path | None = None):
    """Load the LFM3 tokenizer and bind the mask token at id 125033."""
    tok = AutoTokenizer.from_pretrained(path or _ASSETS)
    if tok.convert_tokens_to_ids(MASK_TOKEN) != MASK_TOKEN_ID:
        tok.add_special_tokens({"mask_token": MASK_TOKEN})
    assert tok.mask_token_id == MASK_TOKEN_ID, (
        f"mask token landed at {tok.mask_token_id}, expected {MASK_TOKEN_ID}"
    )
    return tok


__all__ = ["MASK_TOKEN", "MASK_TOKEN_ID", "load_tokenizer"]
