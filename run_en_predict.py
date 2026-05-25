"""
Run EN mDeBERTa predict after training completes.
Saves eng_test_pred_mdeberta.tsv locally.

Run after: run_en_pipeline.py finishes training mdeberta_en_v1.pt
"""

import subprocess, sys
from datetime import datetime
from pathlib import Path

BASE     = Path(__file__).parent
BASELINE = BASE / "BioNNE-R" / "baseline"
LOGFILE  = BASE / "en_predict.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def run(cmd, label, stream=True):
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if stream:
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    log("=== EN mDeBERTa predict ===")

    rc = run([
        "modal", "run", "BioNNE-R/modal_app.py::predict_model",
        "--data-name",  "eng_test_blind.txt",
        "--ckpt-name",  "mdeberta_en_v1.pt",
        "--out-name",   "eng_test_pred_mdeberta.tsv",
        "--batch-size", "64",
        "--max-length", "256",
    ], "predict_en")

    if rc != 0:
        log(f"Predict failed (exit {rc})")
        sys.exit(1)

    dest = str(BASELINE / "outputs" / "eng_test_pred_mdeberta.tsv")
    rc2 = run([
        "modal", "volume", "get", "--force", "bionne-v2",
        "predictions/eng_test_pred_mdeberta.tsv", dest,
    ], "download_en", stream=False)

    if rc2 == 0:
        from pathlib import Path
        n = sum(1 for _ in open(dest)) - 1  # subtract header
        log(f"Downloaded eng_test_pred_mdeberta.tsv → {dest} ({n} predictions)")
    else:
        log(f"Download failed (exit {rc2})")
        sys.exit(1)

    log("=== EN predict done ===")
