"""68 B/token Top-16 teacher logprob cache codec.

Per token, little-endian:
    16 x uint16  low 16 bits of the candidate token ids      (32 B)
    1  x uint16  bitmask of the 17th id bit (bit k <-> id k)  (2 B)
    16 x bf16    teacher log-probabilities                   (32 B)
    1  x bf16    residual tail mass p_T(tau)                  (2 B)

Token ids are lossless over the full vocab range 0..127999 (17 bits);
logprobs and residual mass are bf16-quantized.
"""
import torch

TOP_K = 16
BYTES_PER_TOKEN = 16 * 2 + 2 + 16 * 2 + 2  # 68
VOCAB_SIZE = 128_000


def pack_top16(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    residual_mass: torch.Tensor,
) -> torch.Tensor:
    """Pack Top-16 entries into a contiguous uint8 buffer of shape [N, 68].

    `token_ids` [N,16] integer ids in [0, VOCAB_SIZE), `logprobs` [N,16] and
    `residual_mass` [N] floating point.
    """
    token_ids = token_ids.to(torch.int64)
    if token_ids.ndim != 2 or token_ids.shape[1] != TOP_K:
        raise ValueError(f"token_ids must be [N, {TOP_K}], got {tuple(token_ids.shape)}")
    if token_ids.numel() and (
        token_ids.min() < 0 or token_ids.max() >= VOCAB_SIZE
    ):
        raise ValueError("token id out of range 0..127999")

    low = (token_ids & 0xFFFF).to(torch.uint16)
    high_bits = ((token_ids >> 16) & 1).to(torch.int64)
    weights = (1 << torch.arange(TOP_K, device=token_ids.device))
    high_mask = (high_bits * weights).sum(-1, keepdim=True).to(torch.uint16)

    lp_bf16 = logprobs.to(torch.bfloat16).view(torch.uint16)
    res_bf16 = residual_mass.reshape(-1, 1).to(torch.bfloat16).view(torch.uint16)

    buf = torch.cat(
        [low, high_mask, lp_bf16, res_bf16], dim=-1
    ).contiguous().view(torch.uint8)
    return buf


def unpack_top16(buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of `pack_top16`: returns (token_ids int64 [N,16],
    logprobs float32 [N,16], residual_mass float32 [N])."""
    if buf.ndim != 2 or buf.shape[1] != BYTES_PER_TOKEN:
        raise ValueError(f"buffer must be [N, {BYTES_PER_TOKEN}], got {tuple(buf.shape)}")
    flat = buf.contiguous().view(torch.uint16)

    low = flat[:, :TOP_K].to(torch.int64)
    high_mask = flat[:, TOP_K].to(torch.int64)
    high = ((high_mask.unsqueeze(-1) >> torch.arange(TOP_K, device=buf.device)) & 1) << 16
    token_ids = low | high

    logprobs = flat[:, TOP_K + 1 : TOP_K + 1 + TOP_K].view(torch.bfloat16).float()
    residual = flat[:, TOP_K + 1 + TOP_K].view(torch.bfloat16).float()
    return token_ids, logprobs, residual
