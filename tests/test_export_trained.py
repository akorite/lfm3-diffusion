"""The export path must survive a state dict written by a frozen-MLP trainer."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM  # noqa: E402
from lfm3_dlm.convert import merge_mixer_state  # noqa: E402
from lfm3_dlm.tokenizer import MASK_TOKEN_ID  # noqa: E402
from lfm3_dlm.train.collator import IM_END_ID  # noqa: E402
from lfm3_dlm.train.trainer import DLMTrainer  # noqa: E402


def _cfg():
    return Lfm3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=128, layer_types=["conv", "full_attention"],
                      sliding_window=32, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def test_merge_mixer_state_from_a_frozen_mlp_run(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg()
    base = tmp_path / "base"
    Lfm3ForCausalLM(cfg).save_pretrained(base)
    base_model = Lfm3ForCausalLM.from_pretrained(base)

    trainer = DLMTrainer(base_model, dict(seed=0, s_phys=256, block_size=8,
                                          lr_max=1e-4, lr_min=1e-5, max_steps=10,
                                          warmup_steps=0, w_ar=1.0, freeze_mlps=True,
                                          block_warmup_tokens=0))
    ex = {"prompt_ids": list(range(10, 20)),
          "response_ids": list(range(20, 30)) + [IM_END_ID]}
    trainer.train_step([ex])
    weights = tmp_path / "final-weights.pt"
    torch.save(trainer.model.state_dict(), weights)

    # the trained mask row must survive the export untouched
    trained_mask = trainer.model.state_dict()["model.embed_tokens.weight"][MASK_TOKEN_ID].clone()

    staged = tmp_path / "staged"
    staged.mkdir()
    torch.save(torch.load(weights), staged / "model.bin")
    cfg.save_pretrained(staged)

    out = merge_mixer_state(base, staged, tmp_path / "export")
    exported = Lfm3ForCausalLM.from_pretrained(out)
    assert torch.equal(exported.model.embed_tokens.weight[MASK_TOKEN_ID], trained_mask)
    # frozen MLPs come from the base checkpoint, unfused
    assert hasattr(exported.model.layers[0].feed_forward, "w1")
