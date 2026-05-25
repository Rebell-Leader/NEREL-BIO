"""
Full EN augmentation + re-training pipeline.

Steps:
  1. Run augment_openai.py locally → baseline/data/eng_augmented.txt
  2. Upload augmented file to Modal volume
  3. Concatenate eng_train.txt + eng_augmented.txt on Modal → eng_train_aug.txt
  4. Launch EN mDeBERTa-v3-base training (10 epochs, batch 64, class-weighted)

Run: python run_en_pipeline.py
Log: en_pipeline.log
"""

import subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE      = Path(__file__).parent
BASELINE  = BASE / "BioNNE-R" / "baseline"
LOGFILE   = BASE / "en_pipeline.log"
MODAL_DIR = str(BASE)

AUG_OUT   = str(BASELINE / "data" / "eng_augmented.txt")
TRAIN_TSV = str(BASELINE.parent / "data" / "en" / "train" / "eng-train-rel.tsv")
TEXTS_DIR = str(BASELINE.parent / "data" / "en" / "train" / "texts")


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def run(cmd: list, label: str, stream=True) -> int:
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if stream:
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode


# ── Step 1: Local augmentation ───────────────────────────────────────────────

def step_augment():
    log("=== Step 1: Data augmentation ===")
    aug_path = Path(AUG_OUT)
    if aug_path.exists() and aug_path.stat().st_size > 1000:
        log(f"Augmented file already exists ({aug_path.stat().st_size // 1024} KB), skipping.")
        return True
    rc = run(
        [sys.executable, "-u",
         str(BASELINE / "augment_openai.py"),
         "--train-tsv",   TRAIN_TSV,
         "--train-texts", TEXTS_DIR,
         "--out-jsonl",   AUG_OUT],
        "augment",
    )
    if rc != 0:
        log(f"Augmentation failed (exit {rc})")
        return False
    n = sum(1 for _ in open(AUG_OUT))
    log(f"Augmentation done: {n} examples written to {AUG_OUT}")
    return True


# ── Step 2: Upload to Modal volume ───────────────────────────────────────────

def step_upload():
    log("=== Step 2: Upload augmented data to Modal volume ===")
    rc = run(
        ["modal", "volume", "put", "bionne-v2",
         AUG_OUT, "/data/eng_augmented.txt"],
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
    # Small Modal function to concat two volume files
    concat_script = """
import modal
app = modal.App("concat-aug")
volume = modal.Volume.from_name("bionne-v2")

@app.function(volumes={"/vol": volume}, timeout=120)
def concat():
    from pathlib import Path
    base = Path("/vol/data")
    out  = base / "eng_train_aug.txt"
    with open(out, "wb") as wf:
        for fname in ["eng_train.txt", "eng_augmented.txt"]:
            p = base / fname
            if p.exists():
                wf.write(p.read_bytes())
                print(f"  Added {fname}: {p.stat().st_size // 1024} KB")
            else:
                print(f"  WARNING: {fname} not found")
    n = sum(1 for _ in open(out))
    print(f"eng_train_aug.txt: {n} total instances")
    volume.commit()

@app.local_entrypoint()
def main():
    concat.remote()
"""
    tmp = BASE / "_concat_tmp.py"
    tmp.write_text(concat_script)
    rc = run(["modal", "run", str(tmp)], "concat", stream=True)
    tmp.unlink(missing_ok=True)
    if rc != 0:
        log(f"Concat failed (exit {rc})")
        return False
    log("Concat done.")
    return True


# ── Step 4: Train EN mDeBERTa ────────────────────────────────────────────────

def step_train():
    log("=== Step 4: Launch EN mDeBERTa-v3-base training ===")
    rc = run(
        ["modal", "run", "BioNNE-R/modal_app.py::train_model",
         "--train-data",           "eng_train_aug.txt",
         "--val-data",             "eng_dev.txt",
         "--ckpt-name",            "mdeberta_en_v1.pt",
         "--model-name",           "microsoft/mdeberta-v3-base",
         "--epochs",               "10",
         "--batch-size",           "64",
         "--use-fp16",
         "--gradient-checkpointing"],
        "train_en",
        stream=True,
    )
    if rc == 0:
        log("EN training session complete.")
    else:
        log(f"EN training session exited with code {rc} (likely timeout — re-run to resume).")
    return rc == 0


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=== EN pipeline started ===")
    if not step_augment():
        sys.exit(1)
    if not step_upload():
        sys.exit(1)
    if not step_concat():
        sys.exit(1)
    step_train()  # may time out; orchestrate.py handles resume
    log("=== EN pipeline done (training may need resume runs via orchestrate.py) ===")
