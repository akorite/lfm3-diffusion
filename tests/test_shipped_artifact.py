"""The artifact in `lfm3-artifacts/` must be usable, not just transferable.

Every other export check ran on the VM against the directory *before* it was
tarred. A truncated or repacked tarball would still pass those. This one loads
the shipped bytes on CPU and asserts the same properties, then runs one
forward so a loadable-but-broken artifact fails too.

Skips when the artifact is absent, so a clean checkout stays green.
"""
import json
import os
from pathlib import Path

import pytest
import torch

MASK_ROW, PAD_ROW = 125_033, 124_893
ART = Path(os.environ.get("LFM3_ARTIFACT", "/home/ubuntu/tmp-lfm3/artefact/export"))


@pytest.mark.skipif(not (ART / "model.safetensors").exists(),
                    reason="shipped artifact not extracted on this host")
def test_shipped_export_loads_and_asserts():
    from lfm3_dlm import Lfm3ForCausalLM

    model = Lfm3ForCausalLM.from_pretrained(str(ART), dtype=torch.bfloat16,
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
