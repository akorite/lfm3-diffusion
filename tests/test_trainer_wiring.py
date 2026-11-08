"""The packed-stream contract: the trainer must hand the collator's
metadata to the model, the freeze must hold, and corruption supervision
must never fall back to a zeroed teacher row."""
import pytest
import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.tokenizer import MASK_TOKEN_ID
from lfm3_dlm.train.collator import (
    IM_END_ID, REVEALED_CE_WEIGHT, SharedPrefixCollator,
)
from lfm3_dlm.train.trainer import DLMTrainer

B = 8
S_PHYS = 256


def make_example(P=8, R=12, seed=0, teacher=True, oput_ids=None):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(1000, 60_000, (P,), generator=g)
    resp = torch.randint(1000, 60_000, (R,), generator=g)
    resp[-1] = IM_END_ID
    ex = dict(prompt_ids=prompt.tolist(), response_ids=resp.tolist())
    if teacher:
        ids = torch.randint(1000, 60_000, (16, R), generator=g)
        ids[0] = resp
        ex["teacher_ids"] = ids
        ex["teacher_logprobs"] = torch.log_softmax(torch.randn(16, R, generator=g), dim=0)
        ex["teacher_residual"] = torch.rand(R, generator=g) * 0.02
    if oput_ids is not None:
        ex["oput_ids"] = oput_ids
    return ex


def tiny_model():
    cfg = Lfm3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        layer_types=["conv", "full_attention"],
        sliding_window=32,
    )
    return Lfm3ForCausalLM(cfg)


def tiny_cfg():
    return dict(seed=0, s_phys=S_PHYS, block_size=B, lr_max=1e-4, lr_min=1e-5,
                max_steps=100, warmup_steps=0, w_ar=1.0, freeze_mlps=True,
                block_warmup_tokens=0)


def collate(examples, seed=0):
    rng = torch.Generator().manual_seed(seed)
    return SharedPrefixCollator(s_phys=S_PHYS, block_size=B, rng=rng)(examples)


def test_train_step_hands_packed_metadata_to_the_model():
    """Without doc_ids/stream_ids the diffusion mask silently collapses to
    causal and the noisy streams never see the clean prefix."""
    torch.manual_seed(0)
    model = tiny_model()
    trainer = DLMTrainer(model, tiny_cfg())
    seen = {}

    def hook(_m, _args, kwargs, _out):
        seen.update(kwargs)

    h = model.model.register_forward_hook(hook, with_kwargs=True)
    try:
        trainer.train_step([make_example(seed=5)])
    finally:
        h.remove()

    assert model.config.attention_mode == "block_diffusion"
    assert "stream_ids" in seen and "doc_ids" in seen
    streams = seen["stream_ids"].unique().tolist()
    assert 0 in streams and 1 in streams and 2 in streams


def test_non_mask_embedding_rows_are_bit_identical_after_a_step():
    torch.manual_seed(0)
    model = tiny_model()
    trainer = DLMTrainer(model, tiny_cfg())
    before = model.model.embed_tokens.weight.detach().clone()
    trainer.train_step([make_example(seed=6)])
    after = model.model.embed_tokens.weight.detach()
    assert torch.equal(before[:MASK_TOKEN_ID], after[:MASK_TOKEN_ID])
    assert torch.equal(before[MASK_TOKEN_ID + 1:], after[MASK_TOKEN_ID + 1:])
    assert not torch.equal(before[MASK_TOKEN_ID], after[MASK_TOKEN_ID])


def test_cad_supervision_requires_a_teacher_distribution():
    """Without teacher rows a corrupted slot has no distillation target; it
    must fall back to plain CE rather than a zero-filled teacher row."""
    row = collate([make_example(seed=7, teacher=False)])[0]
    assert int(row.cad_mask.sum()) == 0
    P, R = 8, 16
    resp = torch.tensor(make_example(seed=7, teacher=False)["response_ids"])
    for seg in (slice(P + R, P + 2 * R), slice(P + 2 * R, P + 3 * R)):
        for j in range(len(resp) - 1):
            if float(row.ce_weight[seg.start + j]) == 1.0:
                # selected (corrupted) slot, no teacher: full CE, mask token
                assert row.input_ids[seg.start + j] == MASK_TOKEN_ID
            else:
                assert float(row.ce_weight[seg.start + j]) == pytest.approx(REVEALED_CE_WEIGHT)


def test_oput_ids_are_the_corruption_source():
    """Phase-2 on-manifold corruption draws from the student's own top-1 and
    supervises with CE, never with the teacher cache."""
    oput = torch.arange(2000, 2000 + 12)
    row = collate([make_example(seed=8, oput_ids=oput)])[0]
    assert int(row.cad_mask.sum()) == 0
    P, R = 8, 16
    for seg in (slice(P + R, P + 2 * R), slice(P + 2 * R, P + 3 * R)):
        for j in range(11):
            if float(row.ce_weight[seg.start + j]) == 1.0:
                tok = int(row.input_ids[seg.start + j])
                assert tok == MASK_TOKEN_ID or tok == int(oput[j])


def test_phase_two_oput_branch_corrupts_from_the_student():
    """Past the phase boundary the trainer takes a detached student pass and
    hands the collator its own top-1 as the corruption source."""
    torch.manual_seed(0)
    model = tiny_model()
    cfg = dict(tiny_cfg(), phase1_boundary_tokens=0, oput_ratio=1.0)
    trainer = DLMTrainer(model, cfg)
    assert trainer._oput_due() is True

    ex = make_example(seed=9)
    corrupted = trainer._oput_corrupt([ex])
    oput = corrupted[0]["oput_ids"]
    assert len(oput) == len(ex["response_ids"])

    # the teacher cache is replaced by the student's own prediction
    prompt = torch.tensor([ex["prompt_ids"]])
    resp = torch.tensor([ex["response_ids"]])
    with torch.no_grad():
        hidden = model.model(input_ids=torch.cat([prompt, resp], 1)).last_hidden_state[0]
    pred = hidden[len(ex["prompt_ids"]) - 1:][: len(ex["response_ids"])]
    expect = (pred.float() @ model.get_output_classifier().float().t()).argmax(-1)
    assert torch.equal(oput, expect)

    row = trainer.collator(corrupted)[0]
    assert int(row.cad_mask.sum()) == 0
