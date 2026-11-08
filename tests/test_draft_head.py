"""The draft head must receive the hidden state that predicts the next token.

An identity head - one that returns the model's own next-token distribution as
the proposal - has to be accepted at least once per round. If the plumbing hands
it the wrong slot, acceptance collapses and TPF falls below 1.0, which is
exactly the failure mode a bad draft would also produce. This separates the two.
"""
import torch

from lfm3_dlm.sample import generate

from test_dual_sampler import make_model, greedy_reference


class IdentityHead(torch.nn.Module):
    """Proposes the model's own argmax `k` times."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, hidden, lm_head):
        logits = lm_head(hidden.unsqueeze(0))[0]
        return logits.unsqueeze(0).repeat(self.k, 1)


def _prefix_parity(emitted, reference):
    n = 0
    for a, b in zip(emitted, reference):
        if a != b:
            break
        n += 1
    return n


def test_identity_head_is_accepted_and_keeps_parity():
    model = make_model(["conv", "full_attention"])
    tok = torch.tensor([[5, 9, 13, 21, 34]])
    ref = greedy_reference(model, tok, 12)

    k = 2
    res = generate(model, tok, mode="ar_trust", max_new_tokens=12,
                   block_size=4, small_block=2, isd_width=k,
                   draft_head=IdentityHead(k).eval())
    emitted = res.ids[0, tok.shape[1]:].tolist()[:12]

    assert _prefix_parity(emitted, ref) == 12, "identity draft must not change output"
    # one accepted draft per round means two tokens per two forwards; the tail
    # round can land short, so allow a small margin
    assert float(res.tpf) >= 0.85, (
        f"identity head should be accepted once per round, got tpf={res.tpf:.2f}"
    )


def test_head_receives_the_predictive_slot():
    """The hidden handed to the head must reproduce the verifier's own argmax."""
    model = make_model(["conv", "full_attention"])
    tok = torch.tensor([[5, 9, 13, 21, 34]])
    seen = {}

    class Recorder(torch.nn.Module):
        def forward(self, hidden, lm_head):
            logits = lm_head(hidden.unsqueeze(0))[0]
            seen.setdefault("first", int(logits.argmax()))
            return logits.unsqueeze(0).repeat(2, 1)

    # the head is first consulted after the first token is committed, so the
    # prediction it must reproduce is the *second* token's
    with torch.no_grad():
        first = int(model(input_ids=tok).logits[0, -1, :125_033].argmax())
        cur = torch.cat([tok, torch.tensor([[first]])], 1)
        expected = int(model(input_ids=cur).logits[0, -1, :125_033].argmax())
    generate(model, tok, mode="ar_trust", max_new_tokens=3, block_size=4,
             small_block=2, isd_width=2, draft_head=Recorder().eval())
    assert seen.get("first") == expected
