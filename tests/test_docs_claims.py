"""The model card must not drift from the numbers it claims.

Every figure in `MODEL_CARD.md` and the README's measured table is a value this
repo produced. Those values live in the artifact JSONs under
`lfm3-artifacts/`; if a docs edit or a re-run changes one without the other,
the shipped card silently lies. This test reads both and fails on disagreement.

It skips when the artifacts are absent (they are deliverables, not repo files),
so a clean checkout still runs green.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = Path("/home/ubuntu/lfm3-artifacts")


def _read(name):
    p = ART / name
    if not p.exists():
        pytest.skip(f"{p} not present")
    return p.read_text() if p.suffix == ".md" else json.loads(p.read_text())


def test_evaluation_numbers_match_artifacts():
    trained = _read("heldout_trained.json")
    base = _read("heldout_base.json")
    ppl_t = _read("ppl_trained.json")
    ppl_b = _read("ppl_base.json")

    for name in ("MODEL_CARD.md", "STATUS.md"):
        doc = (ROOT / name).read_text()
        for value in (trained["heldout_loss"], base["heldout_loss"],
                      ppl_t["perplexity"], ppl_b["perplexity"]):
            assert str(value) in doc, f"{value} quoted in the evidence but not {name}"


def test_parity_and_throughput_match_artifacts():
    parity = _read("parity.md")
    card = (ROOT / "MODEL_CARD.md").read_text()

    rows = {m[0]: m[1:] for m in
            re.findall(r"\|\s*(ar_trust_\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
                       parity)}
    assert rows, "no ar_trust rows parsed from parity.md"
    best_parity = rows["ar_trust_4"][0]

    assert best_parity == "1.000", f"parity_sweep now reports {best_parity}"
    card_row = re.search(r"\|\s*\*\*`ar_trust`, width 4\*\*\s*\|[^|]*\|[^|]*\|[^|]*\|\s*\*\*([\d.]+)\*\*", card)
    assert card_row, "the ar_trust row is missing from the performance table"
    assert card_row.group(1) == best_parity, "card parity disagrees with parity.md"


def test_no_unfilled_placeholders_in_user_docs():
    for name in ("MODEL_CARD.md", "README.md", "STATUS.md"):
        text = (ROOT / name).read_text()
        assert "TODO" not in text, f"{name} ships a TODO"
