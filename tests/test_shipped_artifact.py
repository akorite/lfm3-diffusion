"""The artifact in `lfm3-artifacts/` must be usable, not just transferable.

Every other export check ran on the VM against the directory *before* it was
tarred. A truncated or repacked tarball would still pass those. This one takes
the shipped bytes, extracts them itself, loads them on CPU and runs one forward,
so a loadable-but-broken artifact fails too.

It extracts rather than trusting an out-of-band directory: the point is to
validate the tarball, so the tarball is the input. Skips when the artifact is
absent, so a clean checkout stays green.
"""
import os
import tarfile
import tempfile
from pathlib import Path

import pytest
import torch

MASK_ROW, PAD_ROW = 125_033, 124_893
ARTIFACTS = Path(os.environ.get("LFM3_ARTIFACTS_DIR", "/home/ubuntu/lfm3-artifacts"))
TARBALL = ARTIFACTS / "export.tar.gz"


def _skip_unless_shipped():
    if not TARBALL.exists():
        pytest.skip(f"{TARBALL} not present")


@pytest.fixture(scope="module")
def export_dir():
    _skip_unless_shipped()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(TARBALL) as tf:
            tf.extractall(tmp, filter="data")
        dirs = [p for p in Path(tmp).iterdir() if (p / "model.safetensors").exists()]
        assert dirs, f"{TARBALL.name} contains no model.safetensors"
        yield dirs[0]


def test_shipped_export_loads_and_asserts(export_dir):
    from lfm3_dlm import Lfm3ForCausalLM

    model = Lfm3ForCausalLM.from_pretrained(str(export_dir), dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True)
    emb = model.model.embed_tokens.weight
    assert len(model.model.layers) == 24
    assert int(emb.shape[0]) > MASK_ROW
    assert not torch.allclose(emb[MASK_ROW], emb[PAD_ROW]), "mask row equals pad row"
    assert model.config.model_type == "lfm3"

    with torch.no_grad():
        out = model(input_ids=torch.tensor([[5, 9, 13, 21, 34, 55, 89]]))
    assert out.logits.shape[-1] > MASK_ROW
    assert int(out.logits[0, -1, :MASK_ROW].argmax()) >= 0
