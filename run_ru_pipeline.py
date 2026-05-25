"""
RU augmentation + re-training pipeline.

Steps:
  1. Wait for rus_augmented.txt (written by augment_openai.py when all jobs done)
  2. Upload augmented file to Modal volume
  3. Concatenate rus_train.txt + rus_augmented.txt on Modal → rus_train_aug.txt
  4. Create .ckpt sentinel so train_model proceeds
  5. Launch RU mDeBERTa-v3-base training (resume from mdeberta_ru_v2.pt disabled;
     train fresh mdeberta_ru_aug_v1.pt with augmented data)
  6. Predict on rus_test_blind.txt → rus_test_pred_aug.tsv

Run: python run_ru_pipeline.py
Log: ru_pipeline.log
"""

import subprocess, sys, time, json
from datetime import datetime
from pathlib import Path

BASE      = Path(__file__).parent
BASELINE  = BASE / "BioNNE-R" / "baseline"
LOGFILE   = BASE / "ru_pipeline.log"
MODAL_DIR = str(BASE)

AUG_OUT   = str(BASELINE / "data" / "rus_augmented.txt")
TRAIN_TSV = str(BASELINE.parent / "data" / "ru" / "train" / "rus-train-rel.tsv")
TEXTS_DIR = str(BASELINE.parent / "data" / "ru" / "train" / "texts")


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def run(cmd: list, label: str, stream=True) -> int:
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=MODAL_DIR,
    )
    for line in proc.stdout:
        line = line.rstrip()
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if stream:
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode


# ── Step 1: Wait for augmented data ─────────────────────────────────────────

def step_wait_augmented():
    log("=== Step 1: Waiting for rus_augmented.txt ===")
    aug_path = Path(AUG_OUT)
    while True:
        if aug_path.exists() and aug_path.stat().st_size > 1000:
            n = sum(1 for _ in open(aug_path))
            log(f"rus_augmented.txt ready: {n} lines ({aug_path.stat().st_size // 1024} KB)")
            return True
        log(f"Waiting for {aug_path.name}...")
        time.sleep(60)


# ── Step 2: Upload to Modal volume ───────────────────────────────────────────

def step_upload():
    log("=== Step 2: Upload augmented data to Modal volume ===")
    rc = run(
        ["modal", "volume", "put", "bionne-v2",
         AUG_OUT, "/data/rus_augmented.txt"],
        "upload",
        stream=False,
    )
    if rc != 0:
        log(f"Upload failed (exit {rc})")
        return False
    log("Upload done.")
    return True


# ── Step 3: Concatenate on Modal volume ──────────────────────────────────────

def step_concat():
    log("=== Step 3: Concatenate train + augmented on Modal volume ===")
    concat_script = """
import modal
app = modal.App("concat-ru-aug")
volume = modal.Volume.from_name("bionne-v2")

@app.function(volumes={"/vol": volume}, timeout=120)
def concat():
    from pathlib import Path
    import json
    base = Path("/vol/data")
    out  = base / "rus_train_aug.txt"
    with open(out, "wb") as wf:
        for fname in ["rus_train.txt", "rus_augmented.txt"]:
            p = base / fname
            if p.exists():
                wf.write(p.read_bytes())
                print(f"  Added {fname}: {p.stat().st_size // 1024} KB")
            else:
                print(f"  WARNING: {fname} not found")
    n = sum(1 for _ in open(out))
    print(f"rus_train_aug.txt: {n} total instances")
    ckpt = base / "rus_train_aug.txt.ckpt"
    ckpt.write_text(json.dumps({"last_doc_idx": 9999999}))
    volume.commit()

@app.local_entrypoint()
def main():
    concat.remote()
"""
    tmp = BASE / "_concat_ru_tmp.py"
    tmp.write_text(concat_script)
    rc = run(["modal", "run", str(tmp)], "concat", stream=True)
    tmp.unlink(missing_ok=True)
    if rc != 0:
        log(f"Concat failed (exit {rc})")
        return False
    log("Concat done.")
    return True


# ── Step 4: Train RU mDeBERTa ────────────────────────────────────────────────

def step_train():
    log("=== Step 4: Launch RU mDeBERTa-v3-base training (augmented) ===")
    rc = run(
        ["modal", "run", "BioNNE-R/modal_app.py::train_model",
         "--train-data",           "rus_train_aug.txt",
         "--val-data",             "rus_dev.txt",
         "--ckpt-name",            "mdeberta_ru_aug_v1.pt",
         "--model-name",           "microsoft/mdeberta-v3-base",
         "--epochs",               "10",
         "--batch-size",           "128",
         "--use-fp16",
         "--gradient-checkpointing"],
        "train_ru",
        stream=True,
    )
    if rc == 0:
        log("RU training session complete.")
    else:
        log(f"RU training session exited with code {rc} (likely timeout — re-run to resume).")
    return rc == 0


# ── Step 5: Predict on test set ──────────────────────────────────────────────

def step_predict():
    log("=== Step 5: Predict RU test set ===")
    rc = run(
        ["modal", "run", "BioNNE-R/modal_app.py::predict_model",
         "--data-name",   "rus_test_blind.txt",
         "--ckpt-name",   "mdeberta_ru_aug_v1.pt",
         "--out-name",    "rus_test_pred_aug.tsv",
         "--batch-size",  "64"],
        "predict_ru",
        stream=True,
    )
    if rc != 0:
        log(f"Predict failed (exit {rc})")
        return False

    dest = str(BASELINE / "outputs" / "rus_test_pred_aug.tsv")
    rc2 = run(
        ["modal", "volume", "get", "--force", "bionne-v2",
         "predictions/rus_test_pred_aug.tsv", dest],
        "download_ru",
        stream=False,
    )
    if rc2 == 0:
        log(f"Downloaded rus_test_pred_aug.tsv → {dest}")
    else:
        log(f"Download failed (exit {rc2})")
    return rc2 == 0


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=== RU augmented pipeline started ===")
    if not step_wait_augmented():
        sys.exit(1)
    if not step_upload():
        sys.exit(1)
    if not step_concat():
        sys.exit(1)
    if not step_train():
        log("Training may need resume — check ru_pipeline.log and re-run to continue")
    step_predict()
    log("=== RU augmented pipeline done ===")
