"""Block-diffusion masks: logical-position SWA, packed-stream isolation,
train-inference equivalence, and the `attention_mode="causal"` regression."""
import importlib.util
import subprocess
import sys

import pytest
import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.masks import (
    BLOCK_WARMUP_SCHEDULE,
    build_dlm_bool_mask,
    create_dlm_masks_by_layer_type,
    dlm_mask_mod,
)

BLOCK = 32
LAYER_TYPES = [
    "conv", "full_attention", "conv", "sliding_attention",
    "conv", "sliding_attention", "conv", "full_attention",
]


def make_model(attention_mode="block_diffusion", sliding_window=64, layer_types=None):
    cfg = Lfm3Config(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types or LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        sliding_window=sliding_window,
        attention_mode=attention_mode,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=list(layer_types or LAYER_TYPES),
    )
    torch.manual_seed(0)
    return Lfm3ForCausalLM(cfg).eval()


def packed_layout(n_clean_blocks=6, n_noisy_blocks=2, block=BLOCK):
    """One document: stream 0 = P (block 0) + clean blocks, stream 1 = noisy blocks."""
    L0 = block * (1 + n_clean_blocks)
    n_noisy = n_noisy_blocks * block
    position_ids = torch.cat([torch.arange(L0), torch.arange(L0, L0 + n_noisy)]).unsqueeze(0)
    stream_ids = torch.cat([torch.zeros(L0), torch.ones(n_noisy)]).unsqueeze(0).long()
    doc_ids = torch.zeros_like(position_ids)
    return position_ids, stream_ids, doc_ids, L0


def test_dlm_mask_rules_and_mask_mod_consistency():
    position_ids, stream_ids, doc_ids, L0 = packed_layout()
    dense = build_dlm_bool_mask(position_ids, doc_ids, stream_ids, block_size=BLOCK)

    def allowed(q, kv):
        return bool(dense[0, 0, q, kv])

    noisy_q = L0 + 5  # first noisy block (block index L0//BLOCK)
    # bidirectional inside its own block only
    assert all(allowed(L0 + i, noisy_q) for i in range(BLOCK))
    # earlier noisy block is cut
    assert not any(allowed(noisy_q, L0 + BLOCK + i) for i in range(BLOCK))
    # clean prefix blocks are all visible (earlier blocks strictly)
    assert all(allowed(noisy_q, i) for i in range(L0))
    # clean query stays causal and cannot see any noisy column
    assert all(allowed(10, i) for i in range(11))
    assert not any(allowed(10, L0 + i) for i in range(2 * BLOCK))

    mask_mod = dlm_mask_mod(position_ids, doc_ids, stream_ids, block_size=BLOCK)
    for q in (0, L0 // 2, noisy_q, L0 + BLOCK + 3):
        for kv in (0, L0 - 1, L0, L0 + BLOCK + 7, q):
            assert mask_mod(0, 0, q, kv) == bool(dense[0, 0, q, kv])


def test_swa_gates_on_logical_not_physical_index():
    """A noisy block physically far from its clean halo must still see it:
    the window compares logical RoPE positions, not tensor indices."""
    # three packed streams: clean | R_m1 | R_m2 — m2's physical index is one
    # noisy-stream length past its logical position, so context inside the
    # logical window lies physically outside it
    L0 = 224
    n_noisy = 64
    position_ids = torch.cat(
        [torch.arange(L0), torch.arange(L0, L0 + n_noisy), torch.arange(L0, L0 + n_noisy)]
    ).unsqueeze(0)
    stream_ids = torch.cat(
        [torch.zeros(L0), torch.ones(n_noisy), torch.full((n_noisy,), 2)]
    ).unsqueeze(0).long()
    doc_ids = torch.zeros_like(position_ids)
    W = 64
    swa = build_dlm_bool_mask(
        position_ids, doc_ids, stream_ids, block_size=BLOCK, sliding_window=W
    )
    noisy_q = L0 + n_noisy + BLOCK + 5  # m2, second noisy block
    lp_q = L0 + BLOCK + 5
    lp_kv = lp_q - W + 1
    kv_phys = lp_kv  # stream-0 column index
    assert noisy_q - kv_phys > W  # physical distance exceeds the window
    assert bool(swa[0, 0, noisy_q, kv_phys])  # logical distance is inside it
    # and logical distance beyond the window is cut
    assert not bool(swa[0, 0, noisy_q, lp_q - W - 1])
    # same-block noisy pairs stay bidirectional regardless of the window
    assert bool(swa[0, 0, noisy_q, L0 + n_noisy + BLOCK + BLOCK - 1])
    # the other noisy stream is never visible
    assert not any(bool(swa[0, 0, noisy_q, L0 + i]) for i in range(n_noisy))


def test_dlm_mask_dict_has_all_layer_type_keys():
    cfg = make_model().config
    position_ids, stream_ids, doc_ids, _ = packed_layout()
    mapping = create_dlm_masks_by_layer_type(
        cfg,
        attention_mask=torch.ones(1, position_ids.shape[1]),
        position_ids=position_ids,
        doc_ids=doc_ids,
        stream_ids=stream_ids,
        block_size=BLOCK,
        dtype=torch.float32,
    )
    for layer_type in ("full_attention", "sliding_attention", "conv"):
        assert layer_type in mapping
    assert mapping["full_attention"].shape == (1, 1, position_ids.shape[1], position_ids.shape[1])
    assert mapping["conv"].shape == (1, position_ids.shape[1])


def test_block_warmup_schedule():
    assert BLOCK_WARMUP_SCHEDULE == (4, 8, 16, 32)


def test_train_inference_equivalence_bit_identical():
    """Packed `[P | R_clean | R_m1]` noisy block == sequential prefix prefill
    + active block, bitwise, on conv + SWA + full-attention layers — at a
    physical index distance exceeding the sliding window."""
    model = make_model()
    position_ids, stream_ids, doc_ids, L0 = packed_layout(n_clean_blocks=6, n_noisy_blocks=2)
    S = position_ids.shape[1]
    torch.manual_seed(1)
    input_ids = torch.randint(0, 1000, (1, S))

    with torch.no_grad():
        packed = model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            stream_ids=stream_ids,
            doc_ids=doc_ids,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states

        # sequential: clean prefix + first noisy block as its own stream
        seq_ids = torch.cat([input_ids[:, :L0], input_ids[:, L0 : L0 + BLOCK]], dim=1)
        seq_pos = torch.arange(L0 + BLOCK).unsqueeze(0)
        seq_stream = torch.cat([torch.zeros(L0), torch.ones(BLOCK)]).unsqueeze(0).long()
        sequential = model.model(
            input_ids=seq_ids,
            position_ids=seq_pos,
            stream_ids=seq_stream,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states

    for layer_idx, (h_packed, h_seq) in enumerate(zip(packed, sequential)):
        assert torch.equal(
            h_packed[:, :L0], h_seq[:, :L0]
        ), f"clean prefix differs at layer {layer_idx}"
        assert torch.equal(
            h_packed[:, L0 : L0 + BLOCK], h_seq[:, L0:]
        ), f"noisy block differs at layer {layer_idx}"


def test_cross_block_and_cross_document_isolation():
    """Perturbing an earlier noisy block or an adjacent packed document must
    produce zero output delta in the active noisy block and in R_clean."""
    model = make_model()
    position_ids, stream_ids, doc_ids, L0 = packed_layout(n_clean_blocks=6, n_noisy_blocks=2)
    S = position_ids.shape[1]
    torch.manual_seed(1)
    input_ids = torch.randint(0, 1000, (1, S))
    kw = dict(
        position_ids=position_ids,
        stream_ids=stream_ids,
        doc_ids=doc_ids,
        attention_mask=None,
        use_cache=False,
    )
    with torch.no_grad():
        base = model.model(input_ids=input_ids, **kw).last_hidden_state
        perturbed = input_ids.clone()
        perturbed[:, L0 : L0 + 8] = 5  # earlier noisy block columns
        out = model.model(input_ids=perturbed, **kw).last_hidden_state
    assert torch.equal(out[:, :L0], base[:, :L0])  # R_clean untouched
    assert torch.equal(out[:, L0 + BLOCK :], base[:, L0 + BLOCK :])  # active block untouched

    # adjacent packed document: doc 1 shares the row but must not leak into doc 0
    tail = 16
    pos2 = torch.cat([position_ids[0], torch.arange(tail)]).unsqueeze(0)
    stream2 = torch.cat([stream_ids[0], torch.zeros(tail).long()]).unsqueeze(0)
    doc2 = torch.cat([doc_ids[0], doc_ids.new_ones(tail)]).unsqueeze(0)
    ids2 = torch.randint(0, 1000, (1, S + tail))
    kw2 = dict(position_ids=pos2, stream_ids=stream2, doc_ids=doc2, attention_mask=None, use_cache=False)
    with torch.no_grad():
        before = model.model(input_ids=ids2, **kw2).last_hidden_state
        ids2[:, S:] = 7
        after = model.model(input_ids=ids2, **kw2).last_hidden_state
    assert torch.equal(after[:, :S], before[:, :S])


def test_causal_mode_matches_pre_edit_upstream_bitwise(tmp_path):
    """Regression: `attention_mode="causal"` must produce exactly the logits
    the vendored (pre-diffusion) model produced — loaded from the U1 commit."""
    try:
        upstream_src = subprocess.run(
            ["git", "show", "7b6eb46:lfm3_dlm/modeling_lfm3.py"],
            capture_output=True, text=True, check=True, cwd=".",
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git baseline unavailable")
    upstream_src = upstream_src.replace(
        "from .configuration_lfm3", "from lfm3_dlm.configuration_lfm3"
    ).replace("from .lfm3_conv", "from lfm3_dlm.lfm3_conv")
    upstream_path = tmp_path / "upstream_lfm3.py"
    upstream_path.write_text(upstream_src)
    spec = importlib.util.spec_from_file_location("upstream_lfm3", upstream_path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules["upstream_lfm3"] = upstream
    spec.loader.exec_module(upstream)

    cfg = make_model(attention_mode="causal").config
    torch.manual_seed(1)
    current = Lfm3ForCausalLM(cfg).eval()
    upstream_model = upstream.Lfm3ForCausalLM(cfg).eval()
    upstream_model.load_state_dict(current.state_dict())

    torch.manual_seed(2)
    ids = torch.randint(0, 1000, (1, 50))
    packed_ids = torch.cat([ids[:, :20], ids[:, :30]], dim=1)
    packed_pos = torch.cat([torch.arange(20), torch.arange(30)]).unsqueeze(0)
    with torch.no_grad():
        for input_ids, position_ids in ((ids, None), (packed_ids, packed_pos)):
            new = current(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
            old = upstream_model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).logits
            assert torch.equal(new, old), "causal-mode logits diverged from the vendored baseline"
