"""Shared-prefix `[P | R_clean | R_m1 | R_m2]` collator.

Each packed example contributes four physical streams:

* `P` + `R_clean`: prompt then clean response (Dream shift: position at
  logical index `l` predicts clean token `l+1`). Prompt positions carry
  weight 0 except the last (`P-1`), which supervises response token 0;
  `R_clean` position `l >= P` supervises response token `l+1-P`. Noisy
  position `j` supervises response token `j+1`.
* `R_m1` / `R_m2`: noisy copies of the same response under complementary
  Bernoulli(`t`) masks `(m, 1-m)`, `t ~ U[0.35, 0.95]` per document
  (P-POTS stratification). A selected position is replaced by the mask
  token with probability 0.8, else by a non-argmax teacher Top-16 draw
  (0.2). Position `j` supervises response token `j+1`: every response
  token plus the first `im_end` receives CAD supervision exactly once
  across the two noisy streams, while unselected positions carry a
  revealed/CE weight of 0.2. Tail padding (mask token) has weight 0.

`labels[i] = -100` at every stream tail and every packed `doc_id`
boundary so no predictor ever supervises its own target.
"""
from dataclasses import dataclass

import torch

from ..tokenizer import MASK_TOKEN_ID

IM_END_ID = 124_900
T_LOW, T_HIGH = 0.35, 0.95
# t=1.0 rows teach the all-masked corner the sampler's first draft step feeds;
# long training on [0.35, 0.95] alone drifts away from it (see docs/runs).
MASK_TOKEN_FRAC = 0.8
# the sampler feeds the mask embedding at every noisy slot, so a fraction below
# 1.0 leaves a decode-time input the model never saw
REVEALED_CE_WEIGHT = 0.2
TOP_K = 16


@dataclass
class PackedSequence:
    """One physical `[P | R_clean | R_m1 | R_m2 | tail-pad]` row."""

    input_ids: torch.Tensor  # [S] int64
    position_ids: torch.Tensor  # [S] int64, per-doc logical position
    stream_id: torch.Tensor  # [S] 0 clean | 1,2 noisy | 3 tail pad
    doc_id: torch.Tensor  # [S]
    block_id: torch.Tensor  # [S] per-stream-per-doc logical block index
    labels: torch.Tensor  # [S] int64, -100 at tails/doc boundaries
    ce_weight: torch.Tensor  # [S] revealed-position CE weight (0.2)
    ar_weight: torch.Tensor  # [S] clean-stream AR weight
    cad_mask: torch.Tensor  # [S] bool: corrupted positions needing CAD
    teacher_ids: torch.Tensor  # [S, 16] int64 (gathered at cad_mask)
    teacher_logprobs: torch.Tensor  # [S, 16] float32
    teacher_residual: torch.Tensor  # [S] float32
    t: torch.Tensor  # [S] noise level per position

    def to(self, device):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(device))
        return self


def _round_up(n: int, mult: int) -> int:
    return max(mult, -(-n // mult) * mult)


class SharedPrefixCollator:
    def __init__(
        self,
        s_phys: int = 4096,
        block_size: int = 32,
        mask_token_id: int = MASK_TOKEN_ID,
        im_end_id: int = IM_END_ID,
        rng: torch.Generator | None = None,
        n_masks: int = 2,
        t_high: float = T_HIGH,
        mask_token_frac: float = MASK_TOKEN_FRAC,
    ):
        if n_masks < 2:
            raise ValueError("n_masks must be >= 2 (the complementary pair)")
        self.n_masks = n_masks
        self.t_high = t_high
        self.mask_token_frac = mask_token_frac
        self.s_phys = s_phys
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.im_end_id = im_end_id
        self.rng = rng or torch.Generator()

    def __call__(self, examples: list[dict]) -> list[PackedSequence]:
        """Pack examples into as many `s_phys` rows as needed.

        Each example: `prompt_ids` [P], `response_ids` [R] ending in one
        `im_end`; optional `teacher_ids`/`teacher_logprobs` [16, R] and
        `teacher_residual` [R] enabling non-argmax corruption and CAD.
        """
        batches: list[PackedSequence] = []
        cursor = 0
        while cursor < len(examples):
            row, cursor = self._pack_one(examples, cursor)
            batches.append(row)
        return batches

    # ------------------------------------------------------------------
    def _pack_one(self, examples: list[dict], start: int):
        S, B = self.s_phys, self.block_size
        dev = self.rng.device
        input_ids = torch.full((S,), self.mask_token_id, dtype=torch.long, device=dev)
        position_ids = torch.zeros(S, dtype=torch.long, device=dev)
        stream_id = torch.full((S,), 3, dtype=torch.long, device=dev)
        doc_id = torch.zeros(S, dtype=torch.long, device=dev)
        block_id = torch.zeros(S, dtype=torch.long, device=dev)
        labels = torch.full((S,), -100, dtype=torch.long, device=dev)
        ce_weight = torch.zeros(S, device=dev)
        ar_weight = torch.zeros(S, device=dev)
        cad_mask = torch.zeros(S, dtype=torch.bool, device=dev)
        teacher_ids = torch.zeros(S, TOP_K, dtype=torch.long, device=dev)
        teacher_logprobs = torch.zeros(S, TOP_K, device=dev)
        teacher_residual = torch.zeros(S, device=dev)
        t_field = torch.zeros(S, device=dev)

        pos = 0
        i = start
        while i < len(examples):
            ex = examples[i]
            P = len(ex["prompt_ids"])
            R = _round_up(len(ex["response_ids"]), B)
            need = P + (1 + self.n_masks) * R
            if pos + need > S and pos > 0:
                break
            if need > S:
                raise ValueError(
                    f"example {i} needs {need} tokens > s_phys={S}"
                )
            self._write_doc(
                ex, i, pos, P, R,
                input_ids, position_ids, stream_id, doc_id, block_id,
                labels, ce_weight, ar_weight, cad_mask,
                teacher_ids, teacher_logprobs, teacher_residual, t_field,
            )
            pos += need
            i += 1
        return (
            PackedSequence(
                input_ids=input_ids,
                position_ids=position_ids,
                stream_id=stream_id,
                doc_id=doc_id,
                block_id=block_id,
                labels=labels,
                ce_weight=ce_weight,
                ar_weight=ar_weight,
                cad_mask=cad_mask,
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                teacher_residual=teacher_residual,
                t=t_field,
            ),
            i,
        )

    def _write_doc(
        self, ex, doc_idx, pos, P, R,
        input_ids, position_ids, stream_id, doc_id, block_id,
        labels, ce_weight, ar_weight, cad_mask,
        teacher_ids, teacher_logprobs, teacher_residual, t_field,
    ):
        g = self.rng
        B = self.block_size
        dev = g.device
        prompt = torch.as_tensor(ex["prompt_ids"], dtype=torch.long, device=dev)
        resp = torch.as_tensor(ex["response_ids"], dtype=torch.long, device=dev)
        L = P + R  # logical stream length
        e = len(resp) - 1  # index of supervised im_end
        assert resp[e].item() == self.im_end_id, "response must end in im_end"

        # R positions are padded to a multiple of B with the mask token in
        # the response slots of R_clean (logical P+j, j >= len(resp)).
        clean_tokens = torch.cat(
            [prompt, resp, torch.full((R - len(resp),), self.mask_token_id, dtype=torch.long, device=dev)]
        )
        tid = ex.get("teacher_ids")
        tlp = ex.get("teacher_logprobs")
        tres = ex.get("teacher_residual")
        # cache rows arrive on the host; the drawing indices live on the
        # collator's device, so move the teacher tensors once here
        if tid is not None:
            tid, tlp, tres = tid.to(dev), tlp.to(dev), tres.to(dev)
        # phase-2 on-manifold corruption: the student's own top-1 per response
        # position replaces the teacher draw, and CE (not CAD) supervises it
        oid = ex.get("oput_ids")
        cad = tid is not None and oid is None

        # --- noise level, complementary masks over supervised j in [0, e]
        t = T_LOW + (self.t_high - T_LOW) * torch.rand((), generator=g, device=dev).item()
        m = torch.rand(e + 1, generator=g, device=dev) < t  # mask set for stream1

        # corruption draws
        use_mask_tok = torch.rand(e + 1, generator=g, device=dev) < self.mask_token_frac
        if oid is not None:
            confusions = oid[: e + 1].to(dev)
        elif tid is not None:
            cand_k = torch.randint(1, TOP_K, (e + 1,), generator=g, device=dev)
            confusions = tid[cand_k, torch.arange(e + 1, device=dev)]
        else:
            confusions = torch.full((e + 1,), self.mask_token_id, dtype=torch.long, device=dev)
            use_mask_tok = torch.ones(e + 1, dtype=torch.bool, device=dev)
        corrupt_tok = torch.where(use_mask_tok, torch.full_like(confusions, self.mask_token_id), confusions)

        rng_out = {"t": t, "m": m}

        seg0 = slice(pos, pos + L)                    # P + R_clean
        segs = [slice(pos + L + k * R, pos + L + (k + 1) * R)
                for k in range(self.n_masks)]
        for seg in [seg0, *segs]:
            doc_id[seg] = doc_idx
        stream_id[seg0] = 0

        # stream 0: [P | R_clean]
        input_ids[seg0] = clean_tokens
        position_ids[seg0] = torch.arange(L, device=dev)
        block_id[seg0] = torch.arange(L, device=dev) // B
        for l in range(P - 1, P + e):  # clean position l predicts clean l+1
            labels[pos + l] = clean_tokens[l + 1]
            ar_weight[pos + l] = 1.0

        # noisy streams: the complementary pair for K=2, otherwise K
        # independent draws (MMT), each its own stream and therefore isolated
        selections = ((m, ~m) if self.n_masks == 2 else
                      tuple(torch.rand(e + 1, generator=g, device=dev) < t
                            for _ in range(self.n_masks)))
        for stream, (seg, sel) in enumerate(zip(segs, selections), start=1):
            noisy = torch.empty(R, dtype=torch.long, device=dev)
            noisy[: len(resp)] = resp
            noisy[len(resp) :] = self.mask_token_id
            sel_pad = torch.zeros(R, dtype=torch.bool, device=dev)
            sel_pad[: e + 1] = sel
            corrupt_pad = torch.cat([corrupt_tok, torch.full((R - e - 1,), self.mask_token_id, dtype=torch.long, device=dev)])
            noisy = torch.where(sel_pad, corrupt_pad, noisy)

            input_ids[seg] = noisy
            position_ids[seg] = torch.arange(P, P + R, device=dev)
            stream_id[seg] = stream
            block_id[seg] = torch.arange(R, device=dev) // B
            for j in range(R - 1):  # position j predicts response j+1
                if j + 1 <= e:
                    labels[seg.start + j] = resp[j + 1]
            # teacher rows live on predictor positions j in [0, e-1]
            if cad:
                for j in range(R - 1):
                    if j + 1 > e:
                        break
                    teacher_ids[seg.start + j] = tid[:, j + 1]
                    teacher_logprobs[seg.start + j] = tlp[:, j + 1]
                    teacher_residual[seg.start + j] = tres[j + 1]
            for j in range(e):  # j+1 <= e: supervised target exists
                phys = seg.start + j
                if sel[j]:
                    if cad:
                        cad_mask[phys] = True
                    else:
                        # no teacher distribution to distil: plain CE on the
                        # corrupted slot instead of a degenerate CAD target
                        ce_weight[phys] = 1.0
                else:
                    ce_weight[phys] = REVEALED_CE_WEIGHT
            t_field[seg] = t
        self._last_rng = rng_out

    # convenience for tests / trainer introspection
    def last_draws(self):
        return self._last_rng
