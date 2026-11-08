"""Token-parity simulation: AR-Trust greedy replay reaches 100% parity and
realized TPF equals the analytical emitted/steps count."""
import pytest
import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.eval.token_parity import simulate_parity
from lfm3_dlm.sample import generate
from lfm3_dlm.sample.dual_sampler import OUTPUT_VOCAB

VOCAB = 130_000


def make_model(layer_types=("conv", "full_attention",
                            "conv", "sliding_attention"), seed=0):
    torch.manual_seed(seed)
    cfg = Lfm3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512,
        sliding_window=64, layer_types=list(layer_types),
        attention_mode="causal", pad_token_id=0)
    return Lfm3ForCausalLM(cfg).eval()


def greedy_reference(model, prompt_ids, n):
    """Standard B=1 causal greedy AR continuation."""
    cur = prompt_ids.clone()
    out = []
    for _ in range(n):
        tok = int(model(input_ids=cur).logits[0, -1, :OUTPUT_VOCAB].argmax())
        out.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]])], dim=1)
    return out


def test_ar_trust_replay_100_percent_parity_and_exact_tpf():
    """With an oracle draft and the AR continuation forced, every emitted
    token matches AR greedy and realized TPF == emitted / num_steps."""
    model = make_model()
    prompt = torch.randint(0, 100_000, (1, 6))
    continuation = greedy_reference(model, prompt, 12)

    reports = simulate_parity(model, generate, prompt, continuation,
                              strides=(4, 7), taus=(0.55,))
    for rep in reports["ar_trust"]:
        assert rep.token_parity == 1.0
        assert rep.exact_match_steps == len(continuation)
        assert rep.realized_tpf == pytest.approx(rep.expected_tpf)
        assert rep.expected_tpf == pytest.approx(
            rep.n_tokens / rep.n_forwards)
        assert rep.realized_tpf > 1.0          # speculative acceptance works
        assert rep.top1_flip_rate == 0.0
    assert [r.stride for r in reports["ar_trust"]] == [4, 7]


def test_spd_sweep_reports_metrics():
    model = make_model()
    prompt = torch.randint(0, 100_000, (1, 6))
    continuation = greedy_reference(model, prompt, 10)

    reports = simulate_parity(model, generate, prompt, continuation,
                              strides=(6,), taus=(0.30, 0.55, 0.95))
    assert [r.tau for r in reports["spd"]] == [0.30, 0.55, 0.95]
    for rep in reports["spd"]:
        assert 0.0 <= rep.token_parity <= 1.0
        assert rep.n_tokens == len(continuation)
        assert rep.realized_tpf == pytest.approx(
            rep.n_tokens / rep.n_forwards)
        assert rep.tok_per_s > 0.0
