"""
Monitor running jobs and auto-trigger next steps when they complete.

- When RU predict (PID 87927) exits → download rus_test_pred.tsv
- When EN train (PID 88884) exits → launch run_en_predict.py
- When BIL predict (PID 97824) exits → download bilingual_test_pred.tsv

Run: python monitor_completions.py
Log: monitor.log
"""

import subprocess, sys, time, os
from datetime import datetime
from pathlib import Path

BASE     = Path(__file__).parent
BASELINE = BASE / "BioNNE-R" / "baseline"
LOGFILE  = BASE / "monitor.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def run(cmd, label, stream=False):
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(BASE), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if stream:
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode, "\n".join(lines)


RU_PRED_PID  = 106939  # restarted with max_length=256
EN_TRAIN_PID = 88884
BIL_PRED_PID = 102295  # restarted with max_length=192 batch_size=256

ru_done  = False
en_done  = False
bil_done = False

log("=== Completion monitor started ===")
log(f"Watching: RU predict ({RU_PRED_PID}), EN train ({EN_TRAIN_PID}), BIL predict ({BIL_PRED_PID})")

while not (ru_done and en_done and bil_done):
    time.sleep(60)

    if not ru_done and not is_alive(RU_PRED_PID):
        log("RU predict process exited — downloading rus_test_pred.tsv")
        dest = str(BASELINE / "outputs" / "rus_test_pred.tsv")
        rc, out = run([
            "modal", "volume", "get", "--force", "bionne-v2",
            "predictions/rus_test_pred.tsv", dest,
        ], "dl_rus")
        if rc == 0:
            n = sum(1 for _ in open(dest)) - 1
            log(f"Downloaded rus_test_pred.tsv → {dest} ({n} predictions)")
        else:
            log(f"Download failed (rc={rc}): {out[-200:]}")
        ru_done = True

    if not en_done and not is_alive(EN_TRAIN_PID):
        log("EN training process exited — launching run_en_predict.py")
        subprocess.Popen(
            [sys.executable, "-u", str(BASE / "run_en_predict.py")],
            cwd=str(BASE),
            stdout=open(BASE / "en_predict.log", "w"),
            stderr=subprocess.STDOUT,
        )
        log("Launched run_en_predict.py (see en_predict.log)")
        en_done = True

    if not bil_done and not is_alive(BIL_PRED_PID):
        log("BIL predict process exited — downloading bilingual_test_pred.tsv")
        dest = str(BASELINE / "outputs" / "bilingual_test_pred.tsv")
        rc, out = run([
            "modal", "volume", "get", "--force", "bionne-v2",
            "predictions/bilingual_test_pred.tsv", dest,
        ], "dl_bil")
        if rc == 0:
            n = sum(1 for _ in open(dest)) - 1
            log(f"Downloaded bilingual_test_pred.tsv → {dest} ({n} predictions)")
        else:
            log(f"Download failed (rc={rc}): {out[-200:]}")
        bil_done = True

log("=== All monitored jobs complete ===")
