import pytest

from lfm3_dlm import Lfm3Config, load_tokenizer
from lfm3_dlm.tokenizer import MASK_TOKEN, MASK_TOKEN_ID


def test_mask_token_registered_at_125033():
    tok = load_tokenizer()
    assert MASK_TOKEN_ID == 125033
    assert tok.mask_token == MASK_TOKEN
    assert tok.mask_token_id == 125033
    assert "mask_token" in tok.special_tokens_map
    assert tok.encode(MASK_TOKEN, add_special_tokens=False) == [125033]
    assert tok.convert_ids_to_tokens(125033) == MASK_TOKEN


def test_attention_mode_field():
    assert Lfm3Config().attention_mode == "causal"
    assert Lfm3Config(attention_mode="block_diffusion").attention_mode == "block_diffusion"
    with pytest.raises(Exception):
        Lfm3Config(attention_mode="bidirectional")


def test_attention_mode_roundtrip(tmp_path):
    cfg = Lfm3Config(attention_mode="block_diffusion")
    cfg.save_pretrained(tmp_path)
    assert Lfm3Config.from_pretrained(tmp_path).attention_mode == "block_diffusion"
    assert '"attention_mode": "block_diffusion"' in cfg.to_json_string()


def test_vocab_size_stays_128000():
    assert Lfm3Config().vocab_size == 128_000
