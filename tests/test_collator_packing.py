import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.tokenizer import MASK_TOKEN_ID
from lfm3_dlm.train.collator import SharedPrefixCollator, IM_END_ID
from lfm3_dlm.train.trainer import DLMTrainer

B = 8
S_PHYS = 256
IM = IM_END_ID


def make_example(P=10, R=17, lo=1000, hi=60_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(lo, hi, (P,), generator=g)
    resp = torch.randint(lo, hi, (R,), generator=g)
    resp[-1] = IM
    teacher_ids = torch.randint(lo, hi, (16, R), generator=g)
    teacher_ids[0] = resp  # argmax = clean token
    teacher_lp = torch.log_softmax(torch.randn(16, R, generator=g), dim=0)
    teacher_res = torch.rand(R, generator=g) * 0.02
    return dict(
        prompt_ids=prompt.tolist(),
        response_ids=resp.tolist(),
        teacher_ids=teacher_ids,
        teacher_logprobs=teacher_lp,
        teacher_residual=teacher_res,
    )


def collate(examples, seed=0):
    rng = torch.Generator().manual_seed(seed)
    return SharedPrefixCollator(s_phys=S_PHYS, block_size=B, rng=rng)(examples)


def test_segment_layout_and_complement():
    P, Rraw = 10, 17
    R = 24  # round up to B multiple
    ex = make_example(P=P, R=Rraw)
    row = collate([ex])[0]
    resp = torch.tensor(ex["response_ids"])
    L = P + R
    # stream ids
    assert (row.stream_id[:L] == 0).all()
    assert (row.stream_id[L : L + R] == 1).all()
    assert (row.stream_id[L + R : L + 2 * R] == 2).all()
    assert (row.stream_id[L + 2 * R :] == 3).all()
    # logical positions per stream
    assert (row.position_ids[:L] == torch.arange(L)).all()
    assert (row.position_ids[L : L + R] == torch.arange(P, P + R)).all()
    assert (row.position_ids[L + R : L + 2 * R] == torch.arange(P, P + R)).all()
    # complementary corruption: across the two noisy streams each response
    # position is either clean in both or corrupted in exactly one
    n1, n2 = row.input_ids[L : L + R], row.input_ids[L + R : L + 2 * R]
    for j in range(R):
        r = resp[j] if j < Rraw else MASK_TOKEN_ID
        c1, c2 = n1[j].item() != r, n2[j].item() != r
        if j <= Rraw - 1:
            assert c1 != c2 or (not c1 and not c2)  # complement of selected
            if c1 or c2:
                assert c1 != c2  # exactly one stream corrupted
        else:
            assert not c1 and not c2


def test_supervised_exactly_once_across_noisy_streams():
    ex = make_example(P=10, R=17)
    row = collate([ex])[0]
    P, R = 10, 24
    e = 16
    seg1, seg2 = slice(10 + 24, 10 + 48), slice(10 + 48, 10 + 72)
    # cad_mask partitions the supervised predictor positions j in [0, e-1]
    c1 = row.cad_mask[seg1]
    c2 = row.cad_mask[seg2]
    assert (c1 & c2).sum() == 0
    # positions j in [0, e): every supervised predictor is CAD or CE,
    # exactly once as CAD across the two streams
    for j in range(e):
        assert int(row.cad_mask[seg1.start + j]) + int(row.cad_mask[seg2.start + j]) == 1
        # the revealed side gets 0.2 CE
        if row.cad_mask[seg1.start + j]:
            assert row.ce_weight[seg2.start + j] == 0.2
        else:
            assert row.ce_weight[seg1.start + j] == 0.2
    # labels: noisy position j supervises response j+1 -> tokens 1..e
    for j in range(e):
        tgt = ex["response_ids"][j + 1]
        assert row.labels[seg1.start + j] == tgt
        assert row.labels[seg2.start + j] == tgt
    # im_end (token index e) supervised exactly once as CAD target
    sup_e = sum(
        int(row.cad_mask[s.start + e - 1]) for s in (seg1, seg2)
    )
    assert sup_e == 1


def test_no_gradient_to_prompt_or_tail():
    ex = make_example(P=10, R=17)
    row = collate([ex])[0]
    P, R = 10, 24
    # prompt positions except the last (AR shift predictor for r_0) dead
    wsum = row.ce_weight[: P - 1].sum() + row.ar_weight[: P - 1].sum() + row.cad_mask[: P - 1].sum()
    assert wsum == 0
    # mask tail padding (stream 3) and noisy tail pads carry zero weight
    tail = row.stream_id == 3
    assert row.ce_weight[tail].sum() == row.ar_weight[tail].sum() == row.cad_mask[tail].sum() == 0
    e = 16
    for s in (10 + 24, 10 + 48):
        pad = slice(s + e, s + R)  # noisy predictor slots beyond e-1 supervise nothing? j=e..R-2 have label -100
        assert row.ce_weight[pad].sum() == 0
        assert row.cad_mask[pad].sum() == 0
        assert (row.labels[pad] == -100).all()


def test_stream_tails_and_doc_boundaries_are_minus_100():
    exs = [make_example(P=8, R=9, seed=1), make_example(P=6, R=9, seed=2)]
    row = collate(exs)[0]
    # doc0: P+R=8+16=24 -> streams at [0,24),[24,40),[40,56)
    # doc1: P+R=6+16=22 -> [56,78),[78,94),[94,110)
    tails = [23, 39, 55, 77, 93, 109]
    for i in tails:
        assert row.labels[i] == -100, f"tail {i}"
    # no supervised predictor attends its own label: labels are always the
    # NEXT clean token, never the token at the same physical slot of any
    # stream sharing that doc
    sup = (row.ar_weight + row.ce_weight + row.cad_mask.float()) > 0
    for i in sup.nonzero(as_tuple=True)[0].tolist():
        assert row.labels[i] != -100


def test_teacher_rows_align_to_supervised_target():
    ex = make_example(P=10, R=17)
    row = collate([ex])[0]
    seg1 = slice(34, 58)
    for j in range(16):
        assert torch.equal(row.teacher_ids[seg1.start + j], ex["teacher_ids"][:, j + 1])


def test_multi_doc_packing_and_overflow():
    exs = [make_example(P=8, R=9, seed=s) for s in range(20)]
    rows = collate(exs)
    assert len(rows) > 1
    for row in rows:
        # every non-pad position belongs to some doc stream
        assert (row.doc_id[row.stream_id != 3] >= 0).all()


def _tiny_model():
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


def _cfg():
    return dict(
        seed=0, s_phys=S_PHYS, block_size=B, lr_max=1e-4, lr_min=1e-5,
        max_steps=100, warmup_steps=0, w_ar=1.0, freeze_mlps=True,
        block_warmup_tokens=0,
    )


def test_resumable_checkpoint_identical_loss_and_draws(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = DLMTrainer(model, _cfg())
    exs = [make_example(P=8, R=12, seed=11)]

    loss1 = trainer.train_step(exs)
    trainer.save_checkpoint(tmp_path / "ckpt")

    loss2 = trainer.train_step(exs)
    draws2 = trainer.collator.last_draws()

    torch.manual_seed(123)  # perturb global state before reload
    model2 = _tiny_model()
    trainer2 = DLMTrainer(model2, _cfg())
    trainer2.load_checkpoint(tmp_path / "ckpt")
    loss2b = trainer2.train_step(exs)
    draws2b = trainer2.collator.last_draws()

    assert loss2 == loss2b
    assert torch.equal(draws2["m"], draws2b["m"])
    assert draws2["t"] == draws2b["t"]


def test_train_step_updates_only_trainable_params():
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = DLMTrainer(model, _cfg())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainer.train_step([make_example(P=8, R=12, seed=3)])
    changed, frozen = set(), set()
    for n, p in model.named_parameters():
        (changed if not torch.equal(before[n], p.detach()) else frozen).add(n)
    assert any("feed_forward" not in n for n in changed)
    assert all("feed_forward" in n for n in frozen)
