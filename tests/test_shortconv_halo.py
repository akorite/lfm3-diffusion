"""ShortConv clean-halo: gather indices, doc-boundary zeroing, block gating,
and end-to-end equivalence of the packed conv read with sequential decode."""
import torch
import torch.nn.functional as F

from lfm3_dlm.conv_halo import build_conv_halo, halo_conv1d

K = 7
BLOCK = 32


def layout(L0=224, n_noisy=64, block=BLOCK):
    position_ids = torch.cat([torch.arange(L0), torch.arange(L0, L0 + n_noisy)]).unsqueeze(0)
    stream_ids = torch.cat([torch.zeros(L0), torch.ones(n_noisy)]).unsqueeze(0).long()
    doc_ids = torch.zeros_like(position_ids)
    return position_ids, stream_ids, doc_ids


def test_clean_positions_read_physical_neighbors():
    position_ids, stream_ids, doc_ids = layout()
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K, block_size=BLOCK)
    # mid-document clean position: window i-6..i
    assert idx[0, 10].tolist() == [4, 5, 6, 7, 8, 9, 10]
    assert valid[0, 10].all()
    # document start: cross-start columns are invalid (zeroed)
    assert valid[0, 0].tolist() == [False] * 6 + [True]


def test_noisy_block_reads_clean_halo():
    position_ids, stream_ids, doc_ids = layout()
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K, block_size=BLOCK)
    L0 = 224
    # first noisy position: halo = the 6 clean tokens before its block
    assert idx[0, L0].tolist() == [L0 - 6, L0 - 5, L0 - 4, L0 - 3, L0 - 2, L0 - 1, L0]
    assert valid[0, L0].all()
    # later inside the block the window mixes own-block + halo columns
    i = L0 + 3
    assert idx[0, i].tolist() == [L0 - 3, L0 - 2, L0 - 1, L0, L0 + 1, L0 + 2, i]
    assert valid[0, i].all()


def test_noisy_window_never_reads_other_noisy_blocks():
    position_ids, stream_ids, doc_ids = layout()
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K, block_size=BLOCK)
    L0 = 224
    # first noisy position of the second noisy block: the preceding block is
    # noisy too, so no clean tokens exist at those logical positions and the
    # whole halo is invalid (never block 7's physical columns)
    i = L0 + BLOCK
    assert idx[0, i, -1].item() == i
    assert valid[0, i].tolist() == [False] * (K - 1) + [True]
    # none of the valid window columns of a second-block noisy position touch the first noisy block
    noisy_idx = idx[0, L0:]
    noisy_valid = valid[0, L0:]
    forbidden = (noisy_idx >= L0) & (noisy_idx < L0 + BLOCK)
    assert not (forbidden[BLOCK:] & noisy_valid[BLOCK:]).any()


def test_halo_zeroed_across_doc_boundaries():
    position_ids = torch.cat([torch.arange(8), torch.arange(8)]).unsqueeze(0)
    doc_ids = torch.cat([torch.zeros(8), torch.ones(8)]).unsqueeze(0).long()
    stream_ids = torch.zeros_like(position_ids)
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K)
    # second document's first position: every halo column invalid
    assert valid[0, 8].tolist() == [False] * 6 + [True]


def test_halo_conv1d_matches_conv1d_on_clean_sequence():
    torch.manual_seed(0)
    batch, hidden, seq = 2, 8, 40
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    stream_ids = torch.zeros_like(position_ids)
    doc_ids = torch.zeros_like(position_ids)
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K)
    x = torch.randn(batch, hidden, seq)
    weight = torch.randn(hidden, 1, K)
    out = halo_conv1d(x, idx, valid, weight)
    ref = F.conv1d(x, weight, groups=hidden, padding=K - 1)[..., :seq]
    assert torch.allclose(out, ref, atol=1e-6)


def test_halo_conv1d_matches_manual_clean_context_for_noisy():
    torch.manual_seed(0)
    position_ids, stream_ids, doc_ids = layout(L0=64, n_noisy=BLOCK)
    idx, valid = build_conv_halo(position_ids, doc_ids, stream_ids, kernel_size=K, block_size=BLOCK)
    x = torch.randn(1, 8, 96)
    weight = torch.randn(8, 1, K)
    out = halo_conv1d(x, idx, valid, weight)
    # noisy position 64 reads clean columns 58..63 then itself
    expected = (x[0, :, 58:64] * weight[:, 0, :6]).sum(-1) + x[0, :, 64] * weight[:, 0, 6]
    assert torch.allclose(out[0, :, 64], expected)
