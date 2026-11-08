import torch

from lfm3_dlm.data.top16_cache import BYTES_PER_TOKEN, pack_top16, unpack_top16

BOUNDARY_IDS = [0, 65535, 65536, 124900, 125032, 127999]


def make_rows(n):
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 128000, (n, 16), generator=g)
    logprobs = torch.log_softmax(torch.randn(n, 16, generator=g) * 4, dim=-1)
    residual = torch.rand(n, generator=g) * 0.05
    return ids, logprobs, residual


def test_layout_is_68_bytes():
    buf = pack_top16(*make_rows(4))
    assert buf.dtype == torch.uint8
    assert buf.shape == (4, BYTES_PER_TOKEN) == (4, 68)


def test_boundary_ids_roundtrip_bit_identical():
    for tid in BOUNDARY_IDS:
        ids = torch.full((1, 16), tid, dtype=torch.long)
        buf = pack_top16(ids, torch.zeros(1, 16), torch.zeros(1))
        out_ids, _, _ = unpack_top16(buf)
        assert torch.equal(out_ids, ids), f"id {tid} not lossless"


def test_logprobs_roundtrip_bf16():
    ids, logprobs, residual = make_rows(8)
    out_ids, out_lp, out_res = unpack_top16(pack_top16(ids, logprobs, residual))
    assert torch.equal(out_ids, ids)
    assert torch.equal(out_lp.to(torch.bfloat16), logprobs.to(torch.bfloat16))
    assert torch.equal(out_res.to(torch.bfloat16), residual.to(torch.bfloat16))


def test_high_bit_mask_channels():
    # ids straddling the 16-bit boundary per row exercise every mask bit
    ids = torch.arange(16, dtype=torch.long).unsqueeze(0) + 65536
    out_ids, _, _ = unpack_top16(pack_top16(ids, torch.zeros(1, 16), torch.zeros(1)))
    assert torch.equal(out_ids, ids)


def test_bad_shapes_rejected():
    import pytest

    with pytest.raises(ValueError):
        pack_top16(torch.zeros(2, 15, dtype=torch.long), torch.zeros(2, 15), torch.zeros(2))
    with pytest.raises(ValueError):
        pack_top16(torch.full((1, 16), 128000, dtype=torch.long), torch.zeros(1, 16), torch.zeros(1))
    with pytest.raises(ValueError):
        unpack_top16(torch.zeros(3, 67, dtype=torch.uint8))
