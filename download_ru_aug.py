"""Wait for RU augmented predict (PID 120999) to exit, then download."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path

PID  = 120999
DEST = Path("/home/obi/competitions/NEREL-BIO/BioNNE-R/baseline/outputs/rus_test_pred_aug.tsv")
LOG  = Path("/home/obi/competitions/NEREL-BIO/download_ru_aug.log")

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def alive(pid):
    try: os.kill(pid, 0); return True
    except: return False

log(f"Watching PID {PID} for rus_test_pred_aug.tsv ...")
while alive(PID):
    time.sleep(60)

log("Process exited — downloading rus_test_pred_aug.tsv")
r = subprocess.run(
    ["modal", "volume", "get", "--force", "bionne-v2",
     "predictions/rus_test_pred_aug.tsv", str(DEST)],
    capture_output=True, text=True
)
if r.returncode == 0:
    n = sum(1 for _ in open(DEST)) - 1
    log(f"Downloaded → {DEST} ({n} predictions)")
else:
    log(f"Download failed: {r.stdout[-200:]} {r.stderr[-200:]}")
log("Done.")
