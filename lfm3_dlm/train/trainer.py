"""Minimal single-process trainer tying the shared-prefix collator to the
reused-LSE CAD loss. CPU-safe; the G4 path reads `configs/g4_300m.yaml`.

Two phases per the plan: corpus tokens `0..phase1_boundary` use the
teacher-cache corruption collator path; afterwards `oput_ratio` (0.5) of
steps take one detached student OPUT pass for on-manifold corruption.
Checkpoints are written every `checkpoint_interval_tokens` corpus tokens
and are fully resumable (model + optimizer + collator RNG + cursor).
"""
import json
import math
import time
from pathlib import Path

import torch
import yaml

from .collator import SharedPrefixCollator
from .loss_cad import LSE_CHUNK, compute_step_loss, oput_corrupt
from .fp8_mlp import freeze_mlp_weights
from ..masks import BLOCK_WARMUP_SCHEDULE


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t, 1.0)))


def block_size_for_tokens(tokens_seen: int, warmup_tokens: int,
                          stages=BLOCK_WARMUP_SCHEDULE) -> int:
    frac = min(tokens_seen / max(1, warmup_tokens), 1.0)
    idx = min(int(frac * len(stages)), len(stages) - 1)
    return stages[idx]


class DLMTrainer:
    """Ties `SharedPrefixCollator` + `compute_step_loss` + AdamW."""

    def __init__(self, model, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.last_terms: dict = {"l_cad": 0.0, "l_ce": 0.0, "l_ar": 0.0}
        self.tokens_seen = 0
        self.step = 0
        mode = config.get("attention_mode", "block_diffusion")
        if mode not in ("causal", "block_diffusion"):
            raise ValueError(f"unknown attention_mode {mode!r}")
        model.config.attention_mode = mode
        use_flex = bool(config.get("use_flex_attention", False))
        model.config.use_flex_attention = use_flex
        if use_flex:
            # the mask builder emits BlockMasks, which only the flex interface reads
            model.config._attn_implementation = "flex_attention"
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(config.get("seed", 0))
        self.collator = SharedPrefixCollator(
            s_phys=config.get("s_phys", 4096),
            block_size=config.get("block_size", 32),
            rng=self.rng,
            n_masks=config.get("n_masks", 2),
            t_high=config.get("t_high", 0.95),
        )
        if config.get("freeze_mlps", True):
            freeze_mlp_weights(model, fp8=config.get("fp8_mlps", False))
        # frozen embedding rows carry no gradient; decay would still move
        # them, so the tied table sits in a no-decay group
        emb_w = model.get_input_embeddings().weight
        if config.get("freeze_mlps", True):
            groups = [
                {"params": [p for p in model.parameters()
                            if p.requires_grad and p is not emb_w]},
                {"params": [emb_w], "weight_decay": 0.0},
            ]
        else:
            groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
        self.optimizer = torch.optim.AdamW(groups, lr=config["lr_max"])

    # ---------------------------------------------------------------
    def _gather(self, row) -> dict:
        S = row.input_ids.shape[0]
        cad_idx = row.cad_mask.nonzero(as_tuple=True)[0]
        ce_idx = (row.ce_weight > 0).nonzero(as_tuple=True)[0]
        ar_idx = (row.ar_weight > 0).nonzero(as_tuple=True)[0]
        return dict(
            cad_idx=cad_idx,
            teacher_ids=row.teacher_ids[cad_idx],
            teacher_logprobs=row.teacher_logprobs[cad_idx],
            teacher_residual=row.teacher_residual[cad_idx],
            ce_idx=ce_idx,
            ce_targets=row.labels[ce_idx],
            ce_weights=row.ce_weight[ce_idx],
            ar_idx=ar_idx,
            ar_targets=row.labels[ar_idx],
        )

    def train_step(self, examples: list[dict]) -> float:
        """One packed step over a list of prompt/response examples."""
        self.model.train()
        cfg = self.config
        lr = cosine_lr(
            self.step, cfg["max_steps"], cfg.get("warmup_steps", 0),
            cfg["lr_max"], cfg["lr_min"],
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.collator.block_size = block_size_for_tokens(
            self.tokens_seen, cfg.get("block_warmup_tokens", 0),
            tuple(cfg.get("block_stages", BLOCK_WARMUP_SCHEDULE)),
        )
        # the mask reads the annealed size from the model config
        self.model.config.diffusion_block_size = self.collator.block_size
        if self._oput_due():
            examples = self._oput_corrupt(examples)

        rows = [r.to(self.device) for r in self.collator(examples)]
        checkpointed = bool(cfg.get("activation_checkpointing", False))
        total_loss = 0.0
        self.last_terms = {"l_cad": 0.0, "l_ce": 0.0, "l_ar": 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        # one forward per microbatch of packed rows: the rows share a fixed
        # s_phys, and batching them is what keeps the GPU fed
        micro = max(1, int(cfg.get("microbatch", 1)))
        for start in range(0, len(rows), micro):
            chunk = rows[start:start + micro]
            fwd = self._make_forward(chunk, checkpointed)
            out = fwd()
            chunk_losses = []
            for i, row in enumerate(chunk):
                hidden = out.last_hidden_state[i]
                gs = self._gather(row)
                res = compute_step_loss(
                    hidden, self.model.get_output_classifier(),
                    **gs,
                    w_ar=cfg.get("w_ar", 1.0),
                    causal_x0=True,
                )
                chunk_losses.append(res.loss)
                total_loss += float(res.loss.detach())
                self.last_terms["l_cad"] += float(res.l_cad)
                self.last_terms["l_ce"] += float(res.l_ce)
                self.last_terms["l_ar"] += float(res.l_ar)
                self.tokens_seen += int((row.stream_id != 3).sum())
            # one backward per microbatch: the rows share this forward's graph
            torch.stack(chunk_losses).sum().backward()
        self.optimizer.step()
        self.step += 1
        return total_loss

    # ---------------------------------------------------------------
    def _make_forward(self, chunk, checkpointed: bool):
        """Packed forward; `checkpointed` asks the model to recompute layer
        activations instead of retaining them."""
        return lambda: self.model.model(
            input_ids=torch.stack([r.input_ids for r in chunk]),
            position_ids=torch.stack([r.position_ids for r in chunk]),
            doc_ids=torch.stack([r.doc_id for r in chunk]),
            stream_ids=torch.stack([r.stream_id for r in chunk]),
            checkpoint_layers=checkpointed)

    # ---------------------------------------------------------------
    def _oput_due(self) -> bool:
        """Phase-2 gate: past the boundary, `oput_ratio` of steps take a
        detached student pass for on-manifold corruption."""
        cfg = self.config
        ratio = float(cfg.get("oput_ratio", 0.0))
        boundary = cfg.get("phase1_boundary_tokens")
        if ratio <= 0.0 or boundary is None or self.tokens_seen < boundary:
            return False
        return float(torch.rand((), generator=self.rng)) < ratio

    @torch.no_grad()
    def _oput_corrupt(self, examples: list[dict]) -> list[dict]:
        """Attach the student's own top-1 per response position as the
        corruption source (`oput_ids`) for the packed collator."""
        classifier = self.model.get_output_classifier()
        out_examples = []
        for ex in examples:
            prompt = torch.as_tensor(ex["prompt_ids"], dtype=torch.long, device=self.device)
            resp = torch.as_tensor(ex["response_ids"], dtype=torch.long, device=self.device)
            ids = torch.cat([prompt, resp]).unsqueeze(0)
            hidden = self.model.model(input_ids=ids).last_hidden_state[0]
            # position P-1+j predicts response token j
            pred = hidden[len(prompt) - 1 : len(prompt) - 1 + len(resp)]
            ex = dict(ex)
            ex["oput_ids"] = oput_corrupt(pred, classifier, chunk=LSE_CHUNK)
            out_examples.append(ex)
        return out_examples

    # ---------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, prune: int = 0) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rng": self.rng.get_state(),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
            },
            path / "checkpoint.pt",
        )
        (path / "meta.json").write_text(
            json.dumps({"step": self.step, "tokens_seen": self.tokens_seen})
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "checkpoint.pt", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["rng"])
        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]


def train(config_path: str | Path, model=None, data_iter=None) -> DLMTrainer:
    """Entry point: build a trainer from a yaml config and (optionally)
    run a data iterator of example lists."""
    cfg = load_config(config_path)
    if model is None:
        from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM

        model = Lfm3ForCausalLM(Lfm3Config())
    trainer = DLMTrainer(model, cfg)
    if data_iter is not None:
        last_ckpt = 0
        for examples in data_iter:
            trainer.train_step(examples)
            if (
                trainer.tokens_seen - last_ckpt
                >= cfg.get("checkpoint_interval_tokens", 75_000_000)
            ):
                trainer.save_checkpoint(
                    Path(cfg.get("checkpoint_dir", "checkpoints"))
                    / f"step{trainer.step}"
                )
                last_ckpt = trainer.tokens_seen
    return trainer
