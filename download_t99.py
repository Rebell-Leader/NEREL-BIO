"""Auto-download all three t99 predictions when their Modal processes exit."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path

OUTPUTS = Path("/home/obi/competitions/NEREL-BIO/BioNNE-R/baseline/outputs")
LOG = Path("/home/obi/competitions/NEREL-BIO/download_t99.log")

JOBS = {
    "en":  (126081, "predictions/eng_test_pred_t99.tsv",        OUTPUTS / "eng_test_pred_t99.tsv"),
    "ru":  (126116, "predictions/rus_test_pred_t99.tsv",        OUTPUTS / "rus_test_pred_t99.tsv"),
    "bil": (126183, "predictions/bilingual_test_pred_t99.tsv",  OUTPUTS / "bilingual_test_pred_t99.tsv"),
}

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def alive(pid):
    try: os.kill(pid, 0); return True
    except: return False

done = set()
log(f"Watching EN={JOBS['en'][0]}, RU={JOBS['ru'][0]}, BIL={JOBS['bil'][0]}")

while len(done) < 3:
    time.sleep(60)
    for track, (pid, vol_path, dest) in JOBS.items():
        if track in done: continue
        if alive(pid): continue
        log(f"{track.upper()} process exited — downloading {dest.name}")
        r = subprocess.run(
            ["modal", "volume", "get", "--force", "bionne-v2", vol_path, str(dest)],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            n = sum(1 for _ in open(dest)) - 1
            log(f"Downloaded {dest.name} → {n} predictions")
        else:
            log(f"Download FAILED: {r.stderr[-300:]}")
        done.add(track)

log("=== All t99 downloads complete ===")
