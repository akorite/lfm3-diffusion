"""End-to-end G4 driver: teacher cache -> parity gate -> training -> export.

Runs detached on the VM so the pipeline survives harness hiccups. Each stage
appends to /content/PIPELINE.log and a machine-readable /content/PIPELINE.json,
and the last stage writes /content/PIPELINE.done.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/content/repo")
OUT = Path("/content")
STATE = OUT / "PIPELINE.json"
LOG = OUT / "PIPELINE.log"

CACHE_TOKENS = 20_000_000
TRAIN_STEPS = 12_200         # 8192 corpus tokens/step -> ~100M corpus tokens
DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"
CONFIG = "SFT"
SPLIT = "chat"
DATA_FILES = "/content/data/chat.jsonl"   # downloaded once; streaming the Hub stalls
PROMPT_FIELD = "input"
RESPONSE_FIELD = "output"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def record(key, value):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state[key] = value
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATE.write_text(json.dumps(state, indent=1))


def run(cmd, name, timeout=None):
    log(f"start {name}: {cmd}")
    t0 = time.time()
    with tempfile.TemporaryFile(mode="w+") as buf:
        r = subprocess.run(["bash", "-lc", cmd], stdout=buf, stderr=buf,
                           text=True, timeout=timeout)
        buf.seek(0)
        captured = buf.read()
    dt = round(time.time() - t0, 1)
    tail = captured[-800:]
    log(f"end {name} rc={r.returncode} in {dt}s :: {tail}")
    record(name, {"rc": r.returncode, "seconds": dt, "tail": tail})
    return r.returncode


def main():
    log("pipeline start")
    # 1) teacher cache
    meta = OUT / "cache" / "meta.json"
    rc = 0
    cached = json.loads(meta.read_text())["tokens"] if meta.exists() else 0
    if cached >= CACHE_TOKENS:
        log(f"phase0 already complete ({cached} tokens)")
        record("phase0", {"rc": 0, "seconds": 0, "tail": "skipped: cache complete"})
    else:
        rc = run(
            f"cd {ROOT} && python scripts/build_teacher_cache.py "
            f"--data-files {DATA_FILES} "
            f"--prompt-field {PROMPT_FIELD} --response-field {RESPONSE_FIELD} "
            f"--seq-len 4096 --batch-size 16 --head-chunk 256 "
            f"--out /content/cache --max-tokens {CACHE_TOKENS}",
            "phase0")
        if meta.exists():
            record("cache_meta", json.loads(meta.read_text()))
    if rc != 0:
        log("phase0 failed; training would replay a partial cache — aborting")
        record("aborted", "phase0 failed")
        (OUT / "PIPELINE.done").write_text(json.dumps({"error": "phase0 failed"}))
        return

    # 2) parity gate on the teacher cache (cheap, decision-relevant)
    rc = run(f"cd {ROOT} && python - <<'PY'\n"
        "import json, torch\n"
        "from lfm3_dlm import Lfm3ForCausalLM\n"
        "from lfm3_dlm.eval.token_parity import simulate_parity\n"
        "from lfm3_dlm.sample import generate\n"
        "m = Lfm3ForCausalLM.from_pretrained('LiquidAI/LFM3-1.7B', dtype=torch.bfloat16).cuda()\n"
        "p = torch.randint(1000, 60000, (1, 8)).cuda()\n"
        "ref = []\n"
        "cur = p\n"
        "for _ in range(12):\n"
        "    t = int(m(input_ids=cur).logits[0, -1, :125033].argmax()); ref.append(t)\n"
        "    cur = torch.cat([cur, torch.tensor([[t]], device='cuda')], 1)\n"
        "reps = simulate_parity(m, generate, p, ref, strides=(4, 7), taus=(0.55,))\n"
        "best = max(r.realized_tpf for r in reps['spd'])\n"
        "print('PARITY_TPF', round(best, 3))\n"
        "import sys; sys.exit(0 if best >= 4.5 else 1)\n"
        "PY", "parity")

    if rc != 0:
        log("parity stage failed")
        record("aborted", "parity failed")
        (OUT / "PIPELINE.done").write_text(json.dumps({"error": "parity failed"}))
        return

    # 3) training
    run(
        f"cd {ROOT} && python scripts/train_g4.py --cache /content/cache "
        f"--data-files {DATA_FILES} "
        f"--prompt-field {PROMPT_FIELD} --response-field {RESPONSE_FIELD} "
        f"--config configs/g4_run.yaml --out /content/run "
        f"--max-steps {TRAIN_STEPS} --resume",
        "phase1")
    if rc != 0:
        log("phase1 failed")
        (OUT / "PIPELINE.done").write_text(json.dumps({"error": "phase1 failed"}))
        return

    # 4) export + eval
    rc = run(f"cd {ROOT} && python scripts/export_trained.py --run /content/run "
        f"--dst /content/export", "export")
    if rc != 0:
        (OUT / "PIPELINE.done").write_text(json.dumps({"error": "export failed"}))
        return
    run(f"cd {ROOT} && python scripts/eval_trained.py --weights /content/run/final-weights.pt "
        f"--cache /content/cache --out /content/eval.json", "eval")

    log("pipeline done")
    (OUT / "PIPELINE.done").write_text(json.dumps(
        json.loads(STATE.read_text()), indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # keep the failure visible in the artifacts
        import traceback
        log("pipeline crashed: " + traceback.format_exc()[-1200:])
        (OUT / "PIPELINE.done").write_text(json.dumps({"error": str(exc)}))
