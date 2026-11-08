"""Dual-mode sampler tests: AR-Trust parity, cached-vs-uncached steps,
and the SPD <-> AR-Trust `tool_call` switch."""
import torch
import pytest

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.sample import generate
from lfm3_dlm.sample.dual_sampler import (
    IM_END_ID,
    OUTPUT_VOCAB,
    TOOL_CALL_CLOSE_ID,
    TOOL_CALL_OPEN_ID,
    _Sampler,
    build_step_mask,
)
from lfm3_dlm.sample.cache import PrefixCache
from lfm3_dlm.tokenizer import MASK_TOKEN_ID

VOCAB = 130_000


def make_model(layer_types, seed=0, hidden=64, window=64):
    torch.manual_seed(seed)
    cfg = Lfm3Config(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512,
        sliding_window=window, layer_types=layer_types,
        attention_mode="causal", pad_token_id=0,
    )
    return Lfm3ForCausalLM(cfg).eval()


def greedy_reference(model, prompt_ids, n):
    """Standard B=1 causal greedy decoding, recomputed uncached each step."""
    cur = prompt_ids.clone()
    out = []
    for _ in range(n):
        tok = int(model(input_ids=cur).logits[0, -1, :OUTPUT_VOCAB].argmax())
        out.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]])], dim=1)
    return out


@pytest.mark.parametrize(
    "layer_types",
    [["conv", "full_attention"],
     ["full_attention", "conv"],
     ["conv", "full_attention", "conv", "sliding_attention"]],
)
def test_ar_trust_greedy_parity(layer_types):
    """AR-Trust output must be token-for-token identical to causal AR greedy."""
    model = make_model(layer_types)
    ids = torch.randint(0, 100_000, (1, 6))
    res = generate(model, ids, mode="ar_trust", max_new_tokens=14, isd_width=5)
    ref = greedy_reference(model, ids, 14)
    assert res.ids[0, 6:20].tolist() == ref


def test_ar_trust_tpf_above_one_with_matching_drafts():
    """Perfect ISD proposals accept `k` draft + 1 bonus per verify forward."""
    model = make_model(["conv", "full_attention", "conv", "sliding_attention"])
    ids = torch.randint(0, 100_000, (1, 6))
    greedy = greedy_reference(model, ids, 14)

    def oracle(r, gen_pos, emitted):
        return greedy[gen_pos: gen_pos + 8]

    res = generate(model, ids, mode="ar_trust", max_new_tokens=14,
                   isd_width=4, draft_fn=oracle)
    assert res.ids[0, 6:20].tolist() == greedy
    assert res.tpf > 1.0


def test_cached_step_matches_uncached_recompute():
    """Cached B/small_block step logits equal a full-prefix recomputation."""
    P, B = 6, 8
    model = make_model(["conv", "full_attention", "conv", "sliding_attention"])
    ids = torch.randint(0, 100_000, (1, P))

    captured = {}
    def hook(r, phase, feeds, row):
        captured.setdefault("feeds", list(feeds))
        captured.setdefault("logits", sampler._last_logits[0].clone())
        captured.setdefault("committed", sampler._last_committed[0].item())

    with torch.no_grad():
        sampler = _Sampler(model, ids, block_size=B, small_block=4,
                           mode="spd", isd_width=4, draft_fn=None,
                           forced_tokens=None, eos_ids={IM_END_ID},
                           step_hook=hook)
        sampler.limit = B
        sampler._run_spd(B)


    feeds = captured["feeds"]
    cached_logits = captured["logits"]
    C = captured["committed"]

    # uncached recompute: committed prompt + the same fed window, same mask rules
    emb = model.get_input_embeddings()
    e_mask = emb.weight[MASK_TOKEN_ID]
    n = len(feeds)
    extra_emb = torch.stack([
        emb(torch.tensor(t)) if k == "clean" else
        sampler._noisy_embed(sampler.rows[0], p, e_mask)
        for p, k, _b, t in feeds
    ])
    full_emb = torch.cat([emb(ids)[0], extra_emb])
    full_pos = torch.arange(P + n).unsqueeze(0)
    q_pos = kv_pos = full_pos
    q_noisy = torch.zeros(1, P + n, dtype=torch.bool)
    q_noisy[0, P:] = torch.tensor([k == "noisy" for _p, k, _b, _t in feeds])
    q_valid = torch.ones(1, P + n, dtype=torch.bool)
    blocks = torch.zeros(1, P + n, dtype=torch.long)
    blocks[0, P:] = torch.tensor([b for _p, _k, b, _t in feeds])
    kv_block = torch.full((1, P + n), -2**30, dtype=torch.long)
    kv_block[0, P:] = blocks[0, P:]
    kv_fed = torch.zeros(1, P + n, dtype=torch.bool)
    kv_fed[0, P:] = True
    committed = torch.tensor([C])
    masks = {}
    for i, t in enumerate(model.config.layer_types):
        masks[t] = None if t == "conv" else build_step_mask(
            t, model.config.sliding_window if t == "sliding_attention" else None,
            q_pos, q_noisy, q_valid, blocks, kv_pos, kv_block, kv_fed,
            torch.ones(1, P + n, dtype=torch.bool), committed, torch.float32)
    ref = model(inputs_embeds=full_emb.unsqueeze(0), position_ids=full_pos,
                attention_mask=masks).logits[0, :, :OUTPUT_VOCAB]
    assert torch.allclose(cached_logits, ref[P:], atol=1e-4)

def test_tool_call_mode_switch_and_clean_finish():
    """`tool_call` open routes to AR-Trust; close returns to SPD; `im_end`
    terminates without cache corruption."""
    model = make_model(["conv", "full_attention", "conv", "sliding_attention"])
    ids = torch.randint(0, 100_000, (1, 6))
    forced = [1, 2, 3, TOOL_CALL_OPEN_ID, 10, 11, TOOL_CALL_CLOSE_ID, 20, IM_END_ID]
    res = generate(model, ids, mode="hybrid_auto", max_new_tokens=24,
                   block_size=8, small_block=4, forced_tokens=forced)
    modes = [m for _r, _p, m in res.mode_trace]
    assert "ar" in modes and "spd" in modes
    assert modes[0] == "ar" and modes[-1] == "spd"
    tail = res.ids[0, 6:6 + len(forced)].tolist()
    assert tail == forced  # forced commits passed through both modes cleanly


def test_rejects_block_diffusion_mode():
    model = make_model(["conv", "full_attention"])
    model.config.attention_mode = "block_diffusion"
    with pytest.raises(ValueError):
        generate(model, torch.randint(0, 100_000, (1, 4)))
