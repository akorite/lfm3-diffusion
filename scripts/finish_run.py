"""Post-training steps: export, evaluate, and measure parity on the result.

Runs after `train_g4.py` finishes, so the whole run ends in artifacts a reader
can check rather than a metrics file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(name, cmd):
    print(f"start {name}", flush=True)
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-600:]
    print(f"end {name} rc={r.returncode} :: {tail}", flush=True)
    return r.returncode, tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="/content/run3")
    ap.add_argument("--cache", default="/content/cache2")
    ap.add_argument("--export", default="/content/export")
    ap.add_argument("--out", default="/content/FINAL.json")
    args = ap.parse_args(argv)

    weights = Path(args.run) / "final-weights.pt"
    if not weights.exists():
        print(json.dumps({"error": "no trained weights", "path": str(weights)}))
        return 1

    result: dict = {}
    rc, tail = step("export", f"cd {ROOT} && python scripts/export_trained.py "
                              f"--run {args.run} --dst {args.export}")
    result["export"] = {"rc": rc, "tail": tail[-300:]}

    rc, tail = step("eval", f"cd {ROOT} && python scripts/eval_trained.py "
                            f"--weights {weights} --cache {args.cache} "
                            f"--out /content/eval.json --n 8")
    result["eval"] = {"rc": rc, "tail": tail[-300:]}
    if Path("/content/eval.json").exists():
        result["eval_json"] = json.loads(Path("/content/eval.json").read_text())

    rc, tail = step("parity", f"cd {ROOT} && python scripts/parity_sweep.py "
                              f"--weights {weights} --prompts 4 --tokens 24 "
                              f"--out /content/parity.md")
    result["parity"] = {"rc": rc, "tail": tail[-300:]}
    if Path("/content/parity.md").exists():
        result["parity_table"] = Path("/content/parity.md").read_text()

    Path(args.out).write_text(json.dumps(result, indent=1))
    print("FINAL", json.dumps({k: (v if k != "parity_table" else "...") for k, v in result.items()})[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
