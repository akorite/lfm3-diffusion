"""Locked-eval guardrails: decode metadata, ELBO-vs-AR-PPL refusal, and the
XML tool_call / JSON schema validators."""
import pytest
import torch

from lfm3_dlm import Lfm3Config, Lfm3ForCausalLM
from lfm3_dlm.eval.run_eval import run_eval, token_global_mean
from lfm3_dlm.eval.settings import (
    DECODE_CONFIGS, METRIC_FIELDS, compare_metrics, report_metric)
from lfm3_dlm.eval.tool_schema_eval import (
    validate_tool_call_json, validate_tool_call_xml)
from lfm3_dlm.sample import generate

VOCAB = 130_000


def make_model(layer_types=("conv", "full_attention"), seed=0):
    torch.manual_seed(seed)
    cfg = Lfm3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512,
        sliding_window=64, layer_types=list(layer_types),
        attention_mode="causal", pad_token_id=0)
    return Lfm3ForCausalLM(cfg).eval()


# -- guardrails ---------------------------------------------------------------

def test_missing_decode_metadata_raises():
    with pytest.raises(ValueError):
        report_metric("gsm8k.score", 0.8)                      # no decode at all
    with pytest.raises(ValueError):
        report_metric("gsm8k.score", 0.8, decode={"mode": "spd"})  # missing B
    with pytest.raises(ValueError):
        run_eval(make_model(), generate,
                 {"gsm8k": torch.randint(0, 100_000, (1, 4))},
                 decode={"mode": "spd"})                      # missing B/small_block
    with pytest.raises(ValueError):
        run_eval(make_model(), generate,
                 {"not_a_task": torch.randint(0, 100_000, (1, 4))},
                 decode=DECODE_CONFIGS[0])


def test_elbo_vs_ar_ppl_comparison_raises():
    with pytest.raises(ValueError):
        compare_metrics("elbo", -3.1, "ar_ppl", 12.4)
    with pytest.raises(ValueError):
        compare_metrics("ar_perplexity", 10.0, "dlm_ppl", 8.0)
    with pytest.raises(ValueError):
        report_metric("dlm_ppl", 8.0,
                      decode={"mode": "spd", "B": 32, "small_block": 8},
                      context={"baseline_metric": "ar_ppl",
                               "baseline_value": 12.4})


def test_same_family_comparisons_allowed():
    out = compare_metrics("ar_ppl", 11.0, "ar_ppl", 12.0)
    assert out["delta"] == pytest.approx(-1.0)
    out = compare_metrics("gsm8k.score", 0.8, "gsm8k.score", 0.7)
    assert out["rel"] == pytest.approx(0.8 / 0.7 - 1.0)


def test_metric_rows_carry_pinned_fields():
    model = make_model()
    out = run_eval(model, generate,
                   {"gsm8k": torch.randint(0, 100_000, (1, 4))},
                   decode=DECODE_CONFIGS[0], max_new_tokens=8)
    meta = out["metrics"][0]["decode"]
    assert set(meta) == set(METRIC_FIELDS)
    assert meta["mode"] == "spd" and meta["B"] == 32


def test_token_global_mean_weights_by_length():
    losses = [torch.ones(4) * 1.0, torch.ones(2) * 3.0]
    assert token_global_mean(losses) == pytest.approx((4 * 1.0 + 2 * 3.0) / 6)


# -- XML tool_call validator ---------------------------------------------------

VALID_XML = ("<tool_call><name>get_weather</name>"
             "<arguments>{\"city\": \"SF\", \"days\": 3}</arguments>"
             "</tool_call>")


def test_xml_validator_passes_valid_call():
    out = validate_tool_call_xml(VALID_XML)
    assert out["ok"] and out["name"] == "get_weather"
    assert out["arguments"] == {"city": "SF", "days": 3}


@pytest.mark.parametrize("bad", [
    "<tool_call><name>x</name>",                                   # truncated
    "<not_tool><name>x</name><arguments>{}</arguments></not_tool>",# wrong root
    "<tool_call><arguments>{}</arguments></tool_call>",            # missing name
    "<tool_call><name>x</name><arguments>{oops}</arguments></tool_call>",  # bad JSON
    "<tool_call><name>x</name><arguments>[1,2]</arguments></tool_call>",   # non-object
])
def test_xml_validator_catches_malformed(bad):
    out = validate_tool_call_xml(bad)
    assert not out["ok"] and out["errors"]


# -- JSON schema validator -----------------------------------------------------

SCHEMA = {
    "type": "object",
    "required": ["name", "arguments"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "arguments": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
        },
    },
}


def test_json_schema_passes_valid_call():
    payload = {"name": "get_weather", "arguments": {"city": "SF", "days": 3}}
    assert validate_tool_call_json(payload, SCHEMA)["ok"]
    assert validate_tool_call_json('{"name": "f", "arguments": {"city": "X"}}',
                                   SCHEMA)["ok"]


@pytest.mark.parametrize("payload", [
    '{"name": "f"',                                            # unparseable
    {"arguments": {"city": "SF"}},                             # missing required
    {"name": "f", "arguments": {"city": "SF"}, "extra": 1},    # unexpected key
    {"name": "f", "arguments": {"days": 3}},                   # nested missing key
    {"name": "f", "arguments": {"city": "SF", "days": True}},  # bool != integer
    {"name": 7, "arguments": {"city": "SF"}},                  # wrong type
])
def test_json_schema_catches_malformed(payload):
    out = validate_tool_call_json(payload, SCHEMA)
    assert not out["ok"] and out["errors"]
