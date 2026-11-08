import torch
import torch.nn.functional as F

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.modeling_lfm3 import OUTPUT_VOCAB_BOUNDARY
from lfm3_dlm.tokenizer import MASK_TOKEN_ID

# vocab_size is kept just above the mask row so ids stay valid; a 125034x128
# embedding is ~64 MB, cheap for CPU tests.
VOCAB_SIZE = MASK_TOKEN_ID + 1


def synthetic_config(**kwargs):
    return Lfm3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        vocab_size=VOCAB_SIZE,
        layer_types=["conv", "full_attention"],
        sliding_window=64,
        **kwargs,
    )


def test_output_classifier_is_sliced():
    model = Lfm3ForCausalLM(synthetic_config())
    classifier = model.get_output_classifier()
    assert classifier.shape[0] == OUTPUT_VOCAB_BOUNDARY
    assert classifier.shape[0] == MASK_TOKEN_ID
    assert classifier.data_ptr() == model.lm_head.weight.data_ptr()


def test_masked_position_loss_updates_only_mask_row():
    torch.manual_seed(0)
    model = Lfm3ForCausalLM(synthetic_config())
    emb = model.model.embed_tokens.weight

    # Mixer-freeze mode: every embedding row except the mask row is frozen;
    # dead vocab rows (>= MASK_TOKEN_ID) never receive output-side gradient
    # either, because the classifier is sliced.
    frozen_rows = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    frozen_rows[MASK_TOKEN_ID] = False
    emb.register_hook(lambda grad: grad * ~frozen_rows[:, None])

    before = emb[:MASK_TOKEN_ID].detach().clone()

    seq = 16
    input_ids = torch.randint(0, MASK_TOKEN_ID, (1, seq))
    input_ids[0, 8:] = MASK_TOKEN_ID
    out = model.model(input_ids=input_ids)
    hidden = out.last_hidden_state
    logits = F.linear(hidden, model.get_output_classifier())
    assert logits.shape[-1] == MASK_TOKEN_ID

    targets = torch.randint(0, MASK_TOKEN_ID, (1, seq))
    loss = F.cross_entropy(logits[0, 8:], targets[0, 8:])
    loss.backward()

    assert emb.grad is not None
    assert emb.grad[MASK_TOKEN_ID].abs().sum() > 0
    assert emb.grad[:MASK_TOKEN_ID].abs().sum() == 0

    with torch.no_grad():
        emb[MASK_TOKEN_ID] -= 0.1 * emb.grad[MASK_TOKEN_ID]
    assert torch.equal(emb[:MASK_TOKEN_ID], before)


def test_causal_forward_unchanged():
    torch.manual_seed(0)
    model = Lfm3ForCausalLM(synthetic_config())
    model.eval()
    input_ids = torch.randint(0, MASK_TOKEN_ID, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 8, VOCAB_SIZE)
