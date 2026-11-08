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


START_OF_TEXT = "<|startoftext|>"


def format_chat(messages, add_generation_prompt: bool = True) -> str:
    """LFM3's chat format, spelled out because the vendored assets carry no
    chat template. Feeding raw text makes the model repeat the prompt instead
    of answering, so every prompt path goes through here."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    parts = []
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        parts.append(f"{START_OF_TEXT}{m.get('role', 'user')}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append(f"{START_OF_TEXT}assistant\n")
    return "".join(parts)


__all__ = ["MASK_TOKEN", "MASK_TOKEN_ID", "load_tokenizer", "format_chat"]
