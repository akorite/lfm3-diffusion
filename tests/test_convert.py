import hashlib
import json

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.convert import convert_checkpoint, copy_pad_embedding_to_mask, merge_mixer_state
from lfm3_dlm.tokenizer import MASK_TOKEN, MASK_TOKEN_ID

VOCAB_SIZE = MASK_TOKEN_ID + 1
CODE_FILES = ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py", "conv_halo.py", "masks.py")


def synthetic_config(**kwargs):
    return Lfm3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        vocab_size=VOCAB_SIZE,
        layer_types=["conv", "full_attention"],
        sliding_window=64,
        **kwargs,
    )


@pytest.fixture
def source_dir(tmp_path):
    torch.manual_seed(0)
    src = tmp_path / "src"
    model = Lfm3ForCausalLM(synthetic_config()).eval()
    model.save_pretrained(src)
    (src / "LICENSE").write_text("LFM Open License v1.0\n")
    (src / "NOTICE").write_text("Test notice\n")
    return src


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_export_is_self_contained_and_parity(source_dir, tmp_path):
    dst = tmp_path / "out"
    convert_checkpoint(source_dir, dst)

    for f in CODE_FILES:
        assert (dst / f).is_file(), f
    assert (dst / "LICENSE").read_text() == "LFM Open License v1.0\n"
    assert (dst / "NOTICE").is_file()
    assert (dst / "config.json").is_file()
    assert (dst / "model.safetensors").is_file()
    assert (dst / "tokenizer_config.json").is_file()
    assert (dst / "tokenizer.json").is_file()

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["attention_mode"] == "causal"
    assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_lfm3.Lfm3ForCausalLM"

    # Standalone load: custom code resolves from the exported dir itself.
    reloaded = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(dst)
    assert tok.mask_token == MASK_TOKEN and tok.mask_token_id == MASK_TOKEN_ID

    # The exported model embeds the copied mask row.
    emb = reloaded.model.embed_tokens.weight
    pad = reloaded.config.pad_token_id
    assert torch.equal(emb[MASK_TOKEN_ID], emb[pad])
    assert not torch.equal(
        emb[MASK_TOKEN_ID],
        Lfm3ForCausalLM.from_pretrained(source_dir).model.embed_tokens.weight[MASK_TOKEN_ID],
    )


def test_causal_logits_match(source_dir, tmp_path):
    dst = tmp_path / "out"
    convert_checkpoint(source_dir, dst, attention_mode="causal")
    src_model = Lfm3ForCausalLM.from_pretrained(source_dir).eval()
    copy_pad_embedding_to_mask(src_model)  # export copies the pad row; match it here
    reloaded = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True).eval()

    ids = torch.randint(0, 125_000, (2, 33))
    assert torch.allclose(logits(reloaded, ids), logits(src_model, ids), atol=1e-3)


def test_idempotent_and_deterministic(source_dir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    convert_checkpoint(source_dir, a)
    convert_checkpoint(source_dir, b)  # fresh dir
    convert_checkpoint(source_dir, a)  # overwrite existing dir

    def digest(d):
        return hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()

    assert digest(a) == digest(b)
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())


def test_copy_pad_embedding_to_mask():
    model = Lfm3ForCausalLM(synthetic_config())
    emb = model.model.embed_tokens.weight
    pad = model.config.pad_token_id
    with torch.no_grad():
        emb[MASK_TOKEN_ID].zero_()
    copy_pad_embedding_to_mask(model)
    assert torch.equal(emb[MASK_TOKEN_ID], emb[pad])


def test_merge_mixer_state_restores_unfused_layout(source_dir, tmp_path):
    # Simulate a trained checkpoint: perturb mixer weights, fuse w1+w3 -> w13.
    trained = Lfm3ForCausalLM(synthetic_config())
    trained.load_state_dict(Lfm3ForCausalLM.from_pretrained(source_dir).state_dict())
    sd = trained.state_dict()
    with torch.no_grad():
        for k in sd:
            if "self_attn" in k or "conv" in k or "layernorm" in k or "norm" in k:
                sd[k] = sd[k] + 0.5
    fused = {}
    for k, v in sd.items():
        if k.endswith(".feed_forward.w1.weight"):
            base = k[: -len("w1.weight")]
            fused[base + "w13.weight"] = torch.cat([v, sd[base + "w3.weight"]], 0)
        elif k.endswith(".feed_forward.w3.weight"):
            continue
        else:
            fused[k] = v
    trained_dir = tmp_path / "trained"
    trained.save_pretrained(trained_dir, state_dict=fused, safe_serialization=False)

    dst = tmp_path / "merged"
    merge_mixer_state(source_dir, trained_dir, dst, attention_mode="causal")

    merged = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True).eval()
    base = Lfm3ForCausalLM.from_pretrained(source_dir).eval()
    # Mixer weights came from the trained checkpoint ...
    k_attn = "model.layers.1.self_attn.q_proj.weight"
    k_mlp = "model.layers.0.feed_forward.w1.weight"
    assert not torch.equal(merged.state_dict()[k_attn], base.state_dict()[k_attn])
    # ... unfused: w13 was split back to w1/w3 and matches the trained values.
    assert torch.equal(merged.state_dict()[k_mlp], sd["model.layers.0.feed_forward.w1.weight"])
    assert torch.equal(merged.state_dict()["model.layers.0.feed_forward.w3.weight"], sd["model.layers.0.feed_forward.w3.weight"])
    # Frozen position-wise weights came from the base.
    assert torch.equal(merged.state_dict()["model.layers.0.feed_forward.w2.weight"], base.state_dict()["model.layers.0.feed_forward.w2.weight"])
    assert all("w13" not in k for k in merged.state_dict())
