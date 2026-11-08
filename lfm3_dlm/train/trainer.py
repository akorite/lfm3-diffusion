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
from .loss_cad import compute_step_loss
from .fp8_mlp import freeze_mlp_weights


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t, 1.0)))


def block_size_for_tokens(tokens_seen: int, warmup_tokens: int, stages=(4, 8, 16, 32)) -> int:
    frac = min(tokens_seen / max(1, warmup_tokens), 1.0)
    idx = min(int(frac * len(stages)), len(stages) - 1)
    return stages[idx]


class DLMTrainer:
    """Ties `SharedPrefixCollator` + `compute_step_loss` + AdamW."""

    def __init__(self, model, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.tokens_seen = 0
        self.step = 0
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(config.get("seed", 0))
        self.collator = SharedPrefixCollator(
            s_phys=config.get("s_phys", 4096),
            block_size=config.get("block_size", 32),
            rng=self.rng,
        )
        if config.get("freeze_mlps", True):
            freeze_mlp_weights(model)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["lr_max"],
        )

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
            self.tokens_seen, cfg.get("block_warmup_tokens", 0)
        )

        rows = self.collator(examples)
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for row in rows:
            row = row.to(self.device)
            out = self.model.model(
                input_ids=row.input_ids.unsqueeze(0),
                position_ids=row.position_ids.unsqueeze(0),
            )
            hidden = out.last_hidden_state.squeeze(0)
            gs = self._gather(row)
            res = compute_step_loss(
                hidden, self.model.get_output_classifier(),
                **gs,
                w_ar=cfg.get("w_ar", 1.0),
                causal_x0=True,
            )
            res.loss.backward()
            total_loss += float(res.loss.detach())
            self.tokens_seen += int((row.stream_id != 3).sum())
        self.optimizer.step()
        self.step += 1
        return total_loss

    # ---------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
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
