"""The parity metric itself: prefix match against the AR reference."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM  # noqa: E402
from parity_sweep import ar_reference, prefix_parity  # noqa: E402


def test_prefix_parity_counts_the_matching_prefix():
    assert prefix_parity([1, 2, 3, 4], [1, 2, 3, 4])[0] == 1.0
    assert prefix_parity([1, 2, 9, 4], [1, 2, 3, 4]) == (0.5, 2)
    assert prefix_parity([9, 2, 3, 4], [1, 2, 3, 4]) == (0.0, 0)
    assert prefix_parity([], [])[0] == 0.0


def test_ar_reference_is_greedy_and_reproducible():
    torch.manual_seed(0)
    cfg = Lfm3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=128,
                     layer_types=["full_attention", "conv"], pad_token_id=0,
                     bos_token_id=1, eos_token_id=2)
    model = Lfm3ForCausalLM(cfg).eval()
    prompt = torch.randint(10, 250, (1, 5))
    ref = ar_reference(model, prompt, 6)
    assert len(ref) == 6
    assert all(isinstance(t, int) for t in ref)
    assert ar_reference(model, prompt, 6) == ref
    # a longer reference must extend the shorter one: AR decoding is prefix-stable
    assert ar_reference(model, prompt, 3) == ref[:3]
