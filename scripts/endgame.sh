#!/bin/bash
# Endgame for a finished (or cut-short) conversion run: export the checkpoint,
# then measure parity, SPD throughput, and held-out loss on both the trained and
# the base model so the comparison has two numbers rather than one.
#
# Usage: bash scripts/endgame.sh   (expects /content/run4, /content/cache3)
exec > /content/endgame.log 2>&1
set -x
cd /content/repo || exit 1

W=/content/run4/final-weights.pt
if [ ! -f "$W" ]; then
  NEWEST=$(ls -d /content/run4/ckpt-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [ -n "$NEWEST" ]; then
    python - <<PY
import torch
state = torch.load("$NEWEST/checkpoint.pt", map_location="cpu")
torch.save(state["model"], "/content/run4/final-weights.pt")
print("staged final-weights.pt from $NEWEST")
PY
  fi
fi

python scripts/finish_run.py --run /content/run4 --cache /content/cache3 \
  --export /content/export --out /content/FINAL.json

# the export must load as an LFM3 model and keep its mask embedding row
python scripts/assert_export.py --path /content/export | tee /content/export_check.json

# trained measurements
python scripts/heldout_eval.py --weights /content/run4/final-weights.pt \
  --out /content/heldout_trained.json --count 40
python scripts/parity_sweep.py --weights /content/run4/final-weights.pt \
  --prompts 4 --tokens 24 --out /content/parity.md
python scripts/spd_sweep.py --weights /content/run4/final-weights.pt \
  --out /content/spd_sweep.json

# base measurements on the same rows, for the quality comparison
python scripts/heldout_eval.py --out /content/heldout_base.json --count 40

cd /content && tar czf /content/export.tar.gz export 2>/dev/null
echo done > /content/FINISHED
