"""Turn a finished run's artifacts into one markdown table.

Reads what `endgame.sh` produced and prints the evidence for the run's success
criteria, so the report quotes measurements rather than recollection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/content")
    ap.add_argument("--out", default="/content/SUMMARY.md")
    args = ap.parse_args(argv)
    d = Path(args.dir)

    final = _load(str(d / "FINAL.json")) or {}
    trained = _load(str(d / "heldout_trained.json")) or {}
    base = _load(str(d / "heldout_base.json")) or {}
    spd = _load(str(d / "spd_sweep.json")) or {}
    parity_md = (d / "parity.md")
    parity_text = parity_md.read_text() if parity_md.exists() else ""

    lines = ["# Run summary", ""]

    ev = (final.get("eval_json") or {})
    lines += ["## Checkpoint", "",
              f"- weights loaded: {ev.get('weights_keys_loaded')} / {ev.get('weights_keys_total')}",
              f"- tokens per forward: {ev.get('tokens_per_forward')}", ""]
    for s in (ev.get("samples") or [])[:2]:
        lines += [f"> {s[:220]}", ""]

    if trained or base:
        lines += ["## Held-out loss (40 unseen rows)", "",
                  "| model | loss |", "|---|---|",
                  f"| trained | {trained.get('heldout_loss', 'n/a')} |",
                  f"| base | {base.get('heldout_loss', 'n/a')} |", ""]

    if spd:
        best = spd.get("best") or {}
        lines += ["## SPD decode sweep", "",
                  f"- best parity: **{best.get('mean_parity')}** at TPF {best.get('mean_tpf')} "
                  f"(block {best.get('block')}, hard_commit {best.get('hard_commit')})",
                  f"- any config clears 95%: **{spd.get('clears_95')}**", ""]

    if parity_text:
        lines += ["## Parity table (parity_sweep)", "", parity_text.strip(), ""]

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
