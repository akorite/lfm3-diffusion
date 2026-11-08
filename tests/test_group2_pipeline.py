"""End-to-end smoke of the Phase-0 cache -> Phase-1 trainer path.

Runs on CPU with a tiny model so the cache format, the reader and one
training step stay covered without a GPU.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_teacher_cache import teacher_rows  # noqa: E402
from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM  # noqa: E402
from lfm3_dlm.data.top16_cache import BYTES_PER_TOKEN, pack_top16  # noqa: E402
from lfm3_dlm.train.collator import IM_END_ID  # noqa: E402
from lfm3_dlm.train.trainer import DLMTrainer  # noqa: E402
from train_g4 import TeacherCache  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    cfg = Lfm3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1,
                     max_position_embeddings=512, layer_types=["conv", "full_attention"],
                     sliding_window=32)
    return Lfm3ForCausalLM(cfg).eval()


def write_cache(root: Path, model, examples, head_chunk=64):
    root.mkdir(parents=True, exist_ok=True)
    rows, index = [], []
    for prompt_ids, response_ids in examples:
        ids, lps, res = teacher_rows(model, prompt_ids, response_ids, head_chunk)
        index.append({"shard": 0, "offset": len(rows), "P": len(prompt_ids),
                      "R": len(response_ids), "prompt_ids": prompt_ids,
                      "response_ids": response_ids})
        rows.append(pack_top16(ids, lps, res))
    (root / "shard-00000.bin").write_bytes(torch.cat(rows).numpy().tobytes())
    (root / "index.jsonl").write_text("\n".join(json.dumps(e) for e in index) + "\n")
    return root


def test_cache_round_trip_and_one_train_step(tmp_path):
    model = tiny_model()
    prompt = list(range(1000, 1010))
    response = list(range(2000, 2012)) + [IM_END_ID]
    cache = write_cache(tmp_path / "cache", model,
                        [(prompt, response), (prompt[:8], response[:6] + [IM_END_ID])])

    rows = TeacherCache(cache)
    assert len(rows) == 2
    ex = rows.example(0)
    assert ex["prompt_ids"] == prompt and ex["response_ids"] == response
    assert ex["teacher_ids"].shape == (16, len(response))
    assert torch.allclose(ex["teacher_logprobs"].exp().sum(0)
                          + ex["teacher_residual"], torch.ones(len(response)),
                          atol=2e-2)

    cfg = dict(seed=0, s_phys=256, block_size=8, lr_max=1e-4, lr_min=1e-5,
               max_steps=10, warmup_steps=0, w_ar=1.0, freeze_mlps=True,
               block_warmup_tokens=0)
    trainer = DLMTrainer(model, cfg)
    before = model.model.embed_tokens.weight.detach().clone()
    loss = trainer.train_step([rows.example(0), rows.example(1)])
    assert loss > 0
    assert not torch.equal(before, model.model.embed_tokens.weight.detach())


def test_cache_shard_layout_is_68_bytes_per_token(tmp_path):
    model = tiny_model()
    prompt, response = list(range(1000, 1006)), [2000, 2001, 2002, IM_END_ID]
    cache = write_cache(tmp_path / "c", model, [(prompt, response)])
    size = (cache / "shard-00000.bin").stat().st_size
    assert size == BYTES_PER_TOKEN * len(response)
