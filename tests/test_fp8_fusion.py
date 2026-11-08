"""The frozen-MLP wrapper rewrites every SwiGLU weight in the model, so its
forward has to reproduce the plain MLP exactly."""
import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.train.fp8_mlp import FrozenMLP


def tiny_model():
    torch.manual_seed(0)
    cfg = Lfm3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                     num_attention_heads=4, num_key_value_heads=1,
                     max_position_embeddings=256, layer_types=["full_attention"])
    return Lfm3ForCausalLM(cfg)


def test_frozen_mlp_matches_the_plain_mlp():
    """A swapped w1/w3 concat or a wrong split keeps shapes and finiteness, so
    only a numerical check catches it."""
    model = tiny_model()
    mlp = model.model.layers[0].feed_forward
    x = torch.randn(2, 5, 64)
    ref = mlp(x)
    fused = FrozenMLP(mlp, fp8=False)
    assert torch.allclose(fused(x).float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_unfrozen_mlp_receives_weight_gradients():
    """`unfreeze_mlp_weights` is the quality fallback; it is meaningless if
    the custom autograd function still drops the weight gradient."""
    model = tiny_model()
    fused = FrozenMLP(model.model.layers[0].feed_forward, fp8=False)
    for p in fused.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 4, 64, requires_grad=True)
    fused(x).sum().backward()
    assert fused.w13.grad is not None and float(fused.w13.grad.abs().sum()) > 0
    assert torch.allclose(fused.w13.grad, fused.w13_cm.grad.t(), atol=1e-4)
