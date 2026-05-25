"""
Run all missing author baseline (OpenNRE mBERT) predictions, then ensemble.
For each track (eng/rus/bil):
  - 1 deterministic run  (seed=42, stochastic=False)
  - 3 stochastic runs    (seed=1/2/3, stochastic=True)
  - majority-vote ensemble of all 4 → author_{track}_test_ensemble.tsv

EN deterministic is already done; all others are launched in parallel.
"""
import subprocess, threading, sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
LOGFILE = BASE / "author_baseline.log"
MODAL_DIR = str(BASE)

TRACKS = {
    "eng": {"data": "eng_test_blind.txt",        "model_track": "eng"},
    "rus": {"data": "rus_test_blind.txt",         "model_track": "rus"},
    "bil": {"data": "bilingual_test_blind.txt",   "model_track": "bil"},
}

# EN deterministic was correct (7,813 predictions from a prior session).
# RUS deterministic is currently running separately (unsqueeze(1) fix verified at batch 200).
# Everything else was run with broken tokenization → 0 predictions; re-run all.
ALREADY_DONE = {
    "author_eng_test_deterministic.tsv",
    "author_rus_test_deterministic.tsv",
}

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

def run_cmd(cmd: list, label: str) -> int:
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=MODAL_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if any(k in line for k in ("Saved", "batch ", "Error", "error", "Loaded", "Ensemble")):
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode


def predict_one(track_key: str, out_name: str, data: str,
                model_track: str, seed: int, stochastic: bool) -> bool:
    if out_name in ALREADY_DONE:
        log(f"[{track_key}] Skipping {out_name} (already done)")
        return True
    cmd = [
        "modal", "run", "BioNNE-R/modal_app.py::author_baseline_predict",
        "--track", model_track,
        "--data-name", data,
        "--out-name", out_name,
        "--seed", str(seed),
    ]
    if stochastic:
        cmd.append("--stochastic")
    rc = run_cmd(cmd, out_name)
    if rc != 0:
        log(f"[{track_key}] ERROR: {out_name} failed (exit {rc})")
        return False
    log(f"[{track_key}] Done: {out_name}")
    return True


def run_track(track_key: str, data: str, model_track: str):
    jobs = [
        # (out_name, seed, stochastic)
        (f"author_{track_key}_test_deterministic.tsv", 42, False),
        (f"author_{track_key}_test_stochastic_s1.tsv",  1, True),
        (f"author_{track_key}_test_stochastic_s2.tsv",  2, True),
        (f"author_{track_key}_test_stochastic_s3.tsv",  3, True),
    ]
    # Run all 4 in parallel
    results = {}
    threads = []
    def _run(out_name, seed, stochastic):
        results[out_name] = predict_one(track_key, out_name, data, model_track, seed, stochastic)
    for out_name, seed, stoch in jobs:
        t = threading.Thread(target=_run, args=(out_name, seed, stoch), daemon=False)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    if not all(results.values()):
        log(f"[{track_key}] Some predictions failed, skipping ensemble.")
        return

    # Ensemble via track-level wrapper (avoids passing list over CLI)
    ensemble_out = f"author_{track_key}_test_ensemble.tsv"
    log(f"[{track_key}] Running ensemble → {ensemble_out}")
    cmd = [
        "modal", "run", "BioNNE-R/modal_app.py::ensemble_track",
        "--track", track_key,
    ]
    rc = run_cmd(cmd, f"{track_key}_ensemble")
    if rc == 0:
        # Download ensemble result
        dest = str(BASE / "baseline" / "outputs" / ensemble_out)
        dl_cmd = ["modal", "volume", "get", "bionne-v2",
                  f"author_predictions/{ensemble_out}", dest]
        run_cmd(dl_cmd, f"{track_key}_dl")
        log(f"[{track_key}] Saved ensemble to {dest}")
    else:
        log(f"[{track_key}] Ensemble failed (exit {rc})")


if __name__ == "__main__":
    log("=== Author baseline orchestration started ===")
    log("Running all tracks in parallel (EN/RU/BIL × 4 seeds each)")

    track_threads = []
    for track_key, cfg in TRACKS.items():
        t = threading.Thread(
            target=run_track,
            kwargs=dict(track_key=track_key, **cfg),
            daemon=False,
        )
        track_threads.append(t)
        t.start()
    for t in track_threads:
        t.join()

    log("=== Author baseline orchestration complete ===")
