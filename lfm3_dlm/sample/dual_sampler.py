"""Dual-mode cached sampler: `DMax` Soft Parallel Decoding + `AR-Trust` ISD.

SPD (default): denoises one 32-token block at a time, `small_block=8`
sub-block slices once the block is partly committed. Student logits are
SHIFTED BEFORE blending (`z <- cat([z[:, :1], z[:, :-1]])`, so `p_j`
predicts `x_j`, not `x_{j+1}`): the predictor for the token AT position
`p` is the logit emitted by the fed slot at `p - 1`. Masked positions are
re-embedded as `c * E_{v~p}[e(v)] + (1 - c) * e_mask` once `c_j > 0.55`
(`c_j <= 0.30` -> pure `e_mask`). A position hard commits on 2-step top-1
consistency or `c_j > 0.90`; committing the last token of a clean block
grabs token 0 of block `b+1` for free.

AR-Trust (inside `tool_call`, or `mode="ar_trust"`): causal ISD —
`[accepted_clean | MASK x k]` drafts, then the candidates are re-fed
causally and accepted while they match the model argmax. Anchor K/V is
strictly causal, so greedy output is token-for-token identical to B=1 AR
greedy decoding. Draft K/V are never committed: on every rejection the
cache crops back to the accepted length.

EOS is handled per row by `finished_flag` + padding, never post-hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..masks import additive_mask
from ..tokenizer import MASK_TOKEN_ID
from .cache import PrefixCache

TOOL_CALL_OPEN_ID = 125_019
TOOL_CALL_CLOSE_ID = 125_020
IM_END_ID = 124_900
OUTPUT_VOCAB = 125_033

_BLEND_ON = 0.55   # above this, masked slots are re-embedded with the student blend
_HARD_COMMIT = 0.90
_BIG = 2**30


@dataclass
class GenerateResult:
    ids: torch.Tensor            # [batch, prompt + generated] padded with pad_token_id
    tpf: float                   # emitted tokens per model forward
    n_forwards: int
    mode_trace: list = field(default_factory=list)  # [(row, abs_pos, "spd"|"ar")]


@dataclass
class _Row:
    emitted: list = field(default_factory=list)
    committed: int = 0                               # clean KV length (absolute)
    predictors: dict = field(default_factory=dict)   # abs pos -> logit row [V]
    fed_hidden: dict = field(default_factory=dict)   # abs pos -> hidden state
    prev_top1: dict = field(default_factory=dict)    # abs pos -> last step's argmax
    finished: bool = False
    in_tool: bool = False


# ---------------------------------------------------------------------------
# masks


def build_step_mask(
    layer_type: str,
    sliding_window: int | None,
    q_pos: torch.Tensor,      # [B, S] logical positions of the queries
    q_noisy: torch.Tensor,    # [B, S] bool: noisy block-diffusion queries
    q_valid: torch.Tensor,    # [B, S] bool
    q_block: torch.Tensor,    # [B, S] block ids of the queries
    kv_pos: torch.Tensor,     # [B, K] logical positions of the keys
    kv_block: torch.Tensor,   # [B, K] block ids for FED keys (<0 for cache)
    kv_fed: torch.Tensor,     # [B, K] bool: key lives in this step's feed
    kv_valid: torch.Tensor,   # [B, K] bool (False for padded fed slots)
    kv_noisy: torch.Tensor,   # [B, K] bool: fed key is a noisy-block slot
    committed: torch.Tensor,  # [B] clean KV length
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive `[B, 1, S, K]` mask, mirroring `masks.build_dlm_bool_mask`.

    A causal query sees the committed prefix and this step's fed slots up
    to its own position. A noisy query sees clean context only from
    *earlier* blocks — the same rule training applies — plus the noisy
    slots of its own block, bidirectionally. `kv_pos < 0` marks a slot
    that no committed token occupies, so it is never a key.
    """
    ql = q_pos[:, :, None]
    kl = kv_pos[:, None, :]
    qn = q_noisy[:, :, None]
    kn = kv_noisy[:, None, :]
    same_block = q_block[:, :, None] == kv_block[:, None, :]
    fed_ok = kv_fed[:, None, :] & kv_valid[:, None, :]
    cache_ok = (kl >= 0) & (kl < committed[:, None, None])
    visible = (cache_ok | fed_ok) & (kl <= ql)
    valid_q = q_valid[:, :, None]
    earlier_block = kv_block[:, None, :] < q_block[:, :, None]
    own_noisy_block = qn & kn & same_block & fed_ok
    # earlier context is clean by rule: fed noisy slots belong to the current
    # block only, matching masks.py's earlier_clean
    earlier_context = (cache_ok | (fed_ok & ~kn)) & earlier_block
    allowed = valid_q & ((~qn & visible) | (qn & earlier_context)) | (valid_q & own_noisy_block)
    if layer_type == "sliding_attention" and sliding_window is not None:
        allowed &= ((ql - kl) < sliding_window) | own_noisy_block
    return additive_mask(allowed, dtype).unsqueeze(1)


# ---------------------------------------------------------------------------
# sampler


class _Sampler:
    def __init__(self, model, input_ids, block_size, small_block, mode,
                 isd_width, draft_fn, forced_tokens, eos_ids, step_hook,
                 hard_commit=_HARD_COMMIT, draft_head=None):
        self.model = model
        self.cfg = model.config
        self.block = block_size
        self.small = small_block
        self.isd_width = isd_width
        self.draft_fn = draft_fn
        self.draft_head = draft_head
        self._last_hidden = None
        self.forced = list(forced_tokens or [])
        self.eos_ids = set(eos_ids)
        self.step_hook = step_hook
        self.hard_commit = hard_commit

        self.B, self.P = input_ids.shape
        self.device = input_ids.device
        self.cache = PrefixCache(self.cfg)
        self.rows = [_Row() for _ in range(self.B)]
        self.base_mode = "ar" if mode == "ar_trust" else "spd"
        self.row_mode = [self.base_mode] * self.B
        self.auto = mode == "hybrid_auto"
        self.mode_trace = []
        self.n_forwards = 0
        self._bx = {i: [dict() for _ in range(self.B)]
                    for i, t in enumerate(self.cfg.layer_types) if t == "conv"}
        self._hooks = []
        self._fed_logical = [[i for i in range(self.P)] for _ in range(self.B)]
        self._install_bx_hooks()

        want_hidden = self.draft_head is not None
        out = model(input_ids=input_ids, use_cache=True,
                    past_key_values=self.cache.cache,
                    output_hidden_states=want_hidden)
        self.n_forwards += 1
        self.cache.appended = self.P
        self.cache_pos = torch.arange(self.P, device=self.device).expand(
            self.B, self.P).contiguous()
        logits = out.logits[:, -1, :OUTPUT_VOCAB]
        prefill_hidden = None
        if want_hidden and out.hidden_states is not None:
            prefill_hidden = out.hidden_states[-1][:, -1, :].detach()
        for r in range(self.B):
            self.rows[r].committed = self.P
            self.rows[r].predictors[self.P] = logits[r].clone()
            if prefill_hidden is not None:
                self.rows[r].fed_hidden[self.P] = prefill_hidden[r]
        self._fed_logical = None

    # -- Bx recording ---------------------------------------------------------

    def _install_bx_hooks(self):
        """Record `B*x` conv inputs per absolute position so conv states can be
        rebuilt per row from committed tokens only."""
        h = self.cfg.hidden_size
        for i, layer in enumerate(self.model.model.layers):
            if self.cfg.layer_types[i] != "conv":
                continue

            def hook(_m, _in, out, layer_idx=i):
                bx = out[:, :, :h] * out[:, :, 2 * h:]
                for r in range(bx.shape[0]):
                    for j, pos in enumerate(self._fed_logical[r][: bx.shape[1]]):
                        if pos >= 0:
                            self._bx[layer_idx][r][pos] = bx[r, j].detach()
            self._hooks.append(layer.conv.in_proj.register_forward_hook(hook))

    def _set_conv_states(self):
        """Each conv layer's state = last `W` `B*x` of that row's committed range."""
        w = self.cfg.conv_L_cache
        for i, hist in self._bx.items():
            layer = self.cache.cache.layers[i]
            if not layer.has_previous_state.get(0, False):
                continue
            rows_state = []
            for r in range(self.B):
                C = self.rows[r].committed
                cols = [hist[r][p] for p in range(max(0, C - w), C) if p in hist[r]]
                buf = torch.zeros(self.cfg.hidden_size, w,
                                  dtype=layer.conv_states[0].dtype,
                                  device=layer.conv_states[0].device)
                if cols:
                    buf[:, -len(cols):] = torch.stack(cols, dim=-1)
                rows_state.append(buf)
            # assign (not copy_): under past recording the slot accumulates fed
            # history each step, so the committed window must be rebuilt whole
            layer.conv_states[0] = torch.stack(rows_state, dim=0)

    # -- forward --------------------------------------------------------------

    def _forward(self, feeds):
        """Run one cached forward.

        `feeds[r]` = list of `(abs_pos, kind, block_id, tok)`; `kind` is
        `"clean"` (real token, causal), `"noisy"` (blended/mask embed,
        bidirectional within its block), `"mask"` (e_mask, causal) or
        `"pad"`. `tok` overrides the embedding source for clean slots.
        Returns the step's raw logits `[B, S, V]`.
        """
        s_max = max(len(f) for f in feeds)
        for f in feeds:
            f.extend([(-1, "pad", -_BIG, None)] * (s_max - len(f)))

        emb = self.model.get_input_embeddings()
        embeds = torch.zeros(self.B, s_max, self.cfg.hidden_size,
                             dtype=emb.weight.dtype, device=self.device)
        q_pos = torch.zeros(self.B, s_max, dtype=torch.long, device=self.device)
        q_noisy = torch.zeros(self.B, s_max, dtype=torch.bool, device=self.device)
        q_valid = torch.zeros(self.B, s_max, dtype=torch.bool, device=self.device)
        q_block = torch.full((self.B, s_max), -_BIG, dtype=torch.long,
                             device=self.device)
        fed_pos = torch.zeros(self.B, s_max, dtype=torch.long, device=self.device)
        fed_block = torch.full((self.B, s_max), -_BIG, dtype=torch.long,
                               device=self.device)
        fed_valid = torch.zeros(self.B, s_max, dtype=torch.bool, device=self.device)
        fed_noisy = torch.zeros(self.B, s_max, dtype=torch.bool, device=self.device)

        e_mask = emb.weight[MASK_TOKEN_ID]
        self._fed_logical = []
        for r, feed in enumerate(feeds):
            logical = []
            for j, (pos, kind, blk, tok) in enumerate(feed):
                logical.append(pos)
                valid = pos >= 0
                q_pos[r, j] = pos if valid else self.cache.appended + j
                q_noisy[r, j] = kind == "noisy"
                q_valid[r, j] = valid
                q_block[r, j] = blk
                fed_pos[r, j] = q_pos[r, j]
                fed_block[r, j] = blk
                fed_valid[r, j] = valid
                fed_noisy[r, j] = kind == "noisy" and valid
                if not valid or kind == "mask":
                    embeds[r, j] = e_mask
                elif kind == "noisy":
                    embeds[r, j] = self._noisy_embed(self.rows[r], pos, e_mask)
                else:
                    if tok is None:
                        tok = self._token_at(r, pos)
                    embeds[r, j] = emb(torch.tensor([tok], device=self.device))[0]
            self._fed_logical.append(logical)

        committed = torch.tensor([row.committed for row in self.rows],
                                 device=self.device)
        self._last_committed = committed.clone()
        masks = {}
        for i, t in enumerate(self.cfg.layer_types):
            if t == "conv":
                masks[t] = None
                continue
            if t in masks:
                continue  # keyed by layer type: one build serves every layer
            phys = self.cache.cache.layers[i].keys.shape[-2]
            # `phys` is the trailing window on sliding layers, so the cached
            # slots are the tail of the per-row position map
            cp = self.cache_pos[:, self.cache_pos.shape[1] - phys:]
            kv_pos = torch.cat([cp, fed_pos], dim=1)
            kv_block = torch.cat([
                torch.div(cp.clamp(min=0), self.block, rounding_mode="floor"),
                fed_block,
            ], dim=1)
            kv_fed = torch.cat([
                torch.zeros(self.B, phys, dtype=torch.bool, device=self.device),
                torch.ones(self.B, s_max, dtype=torch.bool, device=self.device),
            ], dim=1)
            kv_valid = torch.cat([cp >= 0, fed_valid], dim=1)
            kv_noisy = torch.cat([
                torch.zeros(self.B, phys, dtype=torch.bool, device=self.device),
                fed_noisy,
            ], dim=1)
            masks[t] = build_step_mask(
                t, self.cfg.sliding_window if t == "sliding_attention" else None,
                q_pos, q_noisy, q_valid, q_block,
                kv_pos, kv_block, kv_fed, kv_valid, kv_noisy,
                committed, embeds.dtype)

        self._set_conv_states()
        base_idx = self.cache.appended
        out = self.model.model(inputs_embeds=embeds, position_ids=q_pos,
                               attention_mask=masks,
                               past_key_values=self.cache.cache, use_cache=True)
        hidden = out.last_hidden_state
        out_logits = self.model.lm_head(hidden)
        self.n_forwards += 1
        self.cache.appended += s_max
        for r in range(self.B):
            fresh = fed_pos[r][fed_valid[r]]
            if fresh.numel() and base_idx:
                slot = self.cache_pos[r, :base_idx]
                slot[torch.isin(slot, fresh)] = -1
        self.cache_pos = torch.cat([
            self.cache_pos,
            torch.where(fed_valid, fed_pos,
                        torch.full_like(fed_pos, -1)),
        ], dim=1)

        logits = out_logits[:, :, :OUTPUT_VOCAB]
        self._last_hidden = hidden.detach()
        for r, feed in enumerate(feeds):
            fh = self.rows[r].fed_hidden
            for j, (pos, kind, _b, _t) in enumerate(feed):
                if pos >= 0 and kind in ("clean", "noisy"):
                    fh[pos + 1] = hidden[r, j].detach()
            if len(fh) > 4096:
                for pos in sorted(fh)[:2048]:
                    fh.pop(pos, None)
        # shifted predictors: logits@fed slot pos predict the token at pos+1
        for r, feed in enumerate(feeds):
            for j, (pos, _k, _b, _t) in enumerate(feed):
                if pos >= 0:
                    self.rows[r].predictors[pos + 1] = logits[r, j].clone()
        self._last_logits = logits
        return logits

    def _token_at(self, r, pos):
        row = self.rows[r]
        gen = pos - self.P
        return row.emitted[gen] if 0 <= gen < len(row.emitted) else 0

    def _noisy_embed(self, row, pos, e_mask):
        pred = row.predictors.get(pos)
        if pred is None:
            return e_mask
        p = pred.detach().softmax(-1)
        if float(p.max()) <= _BLEND_ON:
            return e_mask
        return p.max() * (p @ self.model.get_input_embeddings().weight[:OUTPUT_VOCAB]) \
            + (1 - p.max()) * e_mask

    def _crop(self):
        """Drop trailing slots no row still needs. The cache index is not a
        logical position, so a row's committed prefix can end well before
        the last physical slot."""
        committed = torch.tensor([row.committed for row in self.rows],
                                 device=self.device)
        keep = self.cache.appended
        while keep > 0:
            col = self.cache_pos[:, keep - 1]
            if bool((((col >= 0) & (col < committed)).any())):
                break
            keep -= 1
        self.cache.crop(keep)
        self.cache_pos = self.cache_pos[:, :keep]

    # -- emit -----------------------------------------------------------------

    def _pending(self, row):
        """Emitted tokens whose K/V are not committed yet (positions
        `committed .. P+len(emitted)-1`). Derived, never stored — committed
        is the single source of truth."""
        return row.emitted[row.committed - self.P:]

    def _emit(self, r, tok):
        row = self.rows[r]
        if len(row.emitted) >= self.limit:
            row.finished = True
            return tok
        if self.forced:
            tok = self.forced.pop(0)
        row.emitted.append(tok)
        if tok in self.eos_ids:
            row.finished = True
        if self.auto:
            if tok == TOOL_CALL_OPEN_ID and not row.in_tool:
                row.in_tool = True
            elif tok == TOOL_CALL_CLOSE_ID and row.in_tool:
                row.in_tool = False
        # a step can emit several tokens, so record the switch as it happens
        self._refresh_modes()
        return tok

    def _refresh_modes(self):
        for r, row in enumerate(self.rows):
            want = "ar" if (self.base_mode == "ar"
                            or (self.auto and row.in_tool)) else "spd"
            if self.row_mode[r] != want and not row.finished:
                self.mode_trace.append((r, self.P + len(row.emitted), want))
                self.row_mode[r] = want

    # -- SPD ------------------------------------------------------------------

    def _spd_feed(self, r, limit):
        row = self.rows[r]
        P, B = self.P, self.block
        C = row.committed
        pend = self._pending(row)
        n_pend = len(pend)
        block_start = (C // B) * B  # training blocks are logical//B
        block_end = min(block_start + B, P + limit)
        n_noisy = min(block_end - (C + n_pend),
                      B if (C == block_start and n_pend == 0) else self.small)
        if n_pend + n_noisy <= 0:
            return None
        feeds = [(C + i, "clean", (C + i) // B, pend[i])
                 for i in range(n_pend)]
        feeds += [(C + n_pend + i, "noisy", (C + n_pend + i) // B, None)
                  for i in range(n_noisy)]
        return feeds

    def _run_spd(self, limit):
        rows, feeds = [], []
        for r in range(self.B):
            row = self.rows[r]
            f = None
            if (not row.finished and self.row_mode[r] == "spd"
                    and len(row.emitted) < limit):
                f = self._spd_feed(r, limit)
            if f:
                rows.append(r)
                feeds.append(f)
            else:
                feeds.append([])
        if not rows:
            return False
        self._forward(feeds)

        for r in rows:
            row = self.rows[r]
            n_pend = sum(1 for _p, k, _b, _t in feeds[r] if k == "clean")
            row.committed += n_pend

            noisy_pos = [p for p, k, _b, _t in feeds[r] if k == "noisy"]
            commits = []
            for pos in noisy_pos:
                pred = row.predictors[pos]
                top1 = int(pred.argmax())
                conf = float(pred.softmax(-1).max())
                if top1 == row.prev_top1.get(pos) or conf > self.hard_commit:
                    commits.append((pos, top1))
                else:
                    break
            if not commits and noisy_pos:
                pos = noisy_pos[0]
                commits.append((pos, int(row.predictors[pos].argmax())))
            for pos, tok in commits:
                if len(row.emitted) >= limit:
                    break
                row.committed = pos + 1
                self._emit(r, tok)
            row.prev_top1 = {p: int(row.predictors[p].argmax())
                             for p in noisy_pos if p >= row.committed}
            n_em = len(row.emitted)
            if (n_em < limit and not row.finished
                    and n_em % self.block == 0
                    and self.P + n_em in row.predictors):
                self._emit(r, int(row.predictors[self.P + n_em].argmax()))
            if self.step_hook:
                self.step_hook(r, "spd", feeds[r], row)
        self._crop()
        self._refresh_modes()
        return True

    # -- AR-Trust ---------------------------------------------------------------

    def _head_input(self, r, C):
        """Hidden state that predicts the token at `C` (fed slot `C-1`)."""
        return self.rows[r].fed_hidden.get(C)

    def _run_ar(self, limit):
        rows = [r for r in range(self.B)
                if not self.rows[r].finished and self.row_mode[r] == "ar"
                and len(self.rows[r].emitted) < limit]
        if not rows:
            return False
        k = self.isd_width

        for r in rows:
            row = self.rows[r]
            if not self._pending(row):
                pred = row.predictors.get(row.committed)
                assert pred is not None, "AR start needs a stored predictor"
                self._emit(r, int(pred.argmax()))
        # draft forward: [pending | MASK x k] — proposals + commits the pending
        feeds = []
        for r in range(self.B):
            row = self.rows[r]
            pend = self._pending(row)
            if r in rows and pend:
                C = row.committed
                feeds.append(
                    [(C + i, "clean", (C + i) // self.block, pend[i]) for i in range(len(pend))]
                    + [(C + len(pend) + j, "mask", (C + len(pend) + j) // self.block, None) for j in range(k)])
            else:
                feeds.append([])
        self._forward(feeds)
        for r in rows:
            row = self.rows[r]
            row.committed += len(self._pending(row))
        # drop the MASK draft slots so the verify feed's K/V lands at the
        # committed offsets — drafts must never occupy committed positions
        self._crop()

        # candidates: draft_fn proposals or mask-draft argmaxes
        cand = {}
        for r in rows:
            row = self.rows[r]
            if len(row.emitted) >= limit:
                continue
            C = row.committed
            if self.draft_fn is not None:
                cand[r] = list(self.draft_fn(r, C - self.P, row.emitted))[:k]
            elif self.draft_head is not None and self._head_input(r, C) is not None:
                with torch.no_grad():
                    lg = self.draft_head(self._head_input(r, C),
                                         self.model.lm_head)
                n_head = lg.shape[0]
                cand[r] = [int(lg[j, :OUTPUT_VOCAB].argmax())
                           for j in range(min(k, n_head))]
            else:
                # mask-draft proposals: predictor[p] is conditioned on the
                # real prefix + MASKs, i.e. exactly the draft distribution
                cand[r] = [int(row.predictors[C + j].argmax())
                           for j in range(k)]
            cand[r] = (cand[r] + [0] * k)[:k]

        # verify forward: candidates re-fed causally
        feeds = []
        for r in range(self.B):
            if r in cand:
                C = self.rows[r].committed
                feeds.append([(C + j, "clean", (C + j) // self.block, cand[r][j])
                              for j in range(k)])
            else:
                feeds.append([])
        self._forward(feeds)

        for r, cs in cand.items():
            row = self.rows[r]
            C = row.committed
            m = k
            for j in range(k):
                if int(row.predictors[C + j].argmax()) != cs[j]:
                    m = j
                    break
            # accepted candidates' causal K/V already sits in the cache; the
            # first-mismatch correction is committed as pending next round
            for tok in cs[:m]:
                self._emit(r, tok)
            row.committed = C + m
            self._emit(r, int(row.predictors[C + m].argmax()))
            if self.step_hook:
                self.step_hook(r, "ar_verify", feeds[r], row)
        self._crop()
        self._refresh_modes()
        return True

    # -- driver ------------------------------------------------------------------

    def run(self, max_new_tokens):
        limit = self.limit = max_new_tokens
        while True:
            if not any(not row.finished and len(row.emitted) < limit
                       for row in self.rows):
                break
            progressed = self._run_ar(limit) | self._run_spd(limit)
            if not progressed:
                break
        pad = self.cfg.pad_token_id or 0
        ids = torch.full((self.B, self.P + limit), pad, dtype=torch.long,
                         device=self.device)
        ids[:, : self.P] = self._input_ids
        for r, row in enumerate(self.rows):
            for i, tok in enumerate(row.emitted):
                ids[r, self.P + i] = tok
        emitted = sum(len(r.emitted) for r in self.rows)
        return GenerateResult(ids=ids,
                              tpf=emitted / max(self.n_forwards, 1),
                              n_forwards=self.n_forwards,
                              mode_trace=self.mode_trace)


def generate(model, input_ids, mode="hybrid_auto", max_new_tokens=64,
             block_size=32, small_block=8, isd_width=6, draft_fn=None,
             forced_tokens=None, eos_token_ids=None, step_hook=None,
             hard_commit=_HARD_COMMIT, draft_head=None):
    """Dual-mode generation entry point (CPU-safe, no CUDA dependency).

    `mode` is `"spd"`, `"ar_trust"` or `"hybrid_auto"` (SPD that switches to
    AR-Trust between `tool_call` open/close). `draft_fn(r, gen_pos, emitted)`
    may supply ISD candidates; `forced_tokens` overrides emitted values (the
    commit machinery still runs) for tests; `step_hook(r, phase, feeds, row)`
    observes each step.
    """
    if mode not in ("spd", "ar_trust", "hybrid_auto"):
        raise ValueError(f"unknown mode {mode!r}")
    if model.config.attention_mode != "causal":
        raise ValueError("the cached sampler drives the causal clean stream; "
                         "set attention_mode='causal'")
    if input_ids.shape[0] > 1 and "sliding_attention" in model.config.layer_types:
        raise ValueError("batched decoding shares one sliding-window ring, "
                         "so a lagging row would lose its own window")
    with torch.inference_mode():
        eos = eos_token_ids if eos_token_ids is not None else (IM_END_ID,)
        sampler = _Sampler(model, input_ids, block_size, small_block, mode,
                           isd_width, draft_fn, forced_tokens, eos, step_hook,
                           hard_commit=hard_commit, draft_head=draft_head)
        sampler._input_ids = input_ids
        try:
            return sampler.run(max_new_tokens)
        finally:
            for h in sampler._hooks:
                h.remove()





__all__ = ["GenerateResult", "generate", "build_step_mask",
           "TOOL_CALL_OPEN_ID", "TOOL_CALL_CLOSE_ID", "IM_END_ID"]
