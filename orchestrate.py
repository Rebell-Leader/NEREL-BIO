"""
Auto-resume RU and BIL mDeBERTa training, then predict on test sets.
Run: python orchestrate.py [--ru-pid PID] [--bil-pid PID]
Logs to: orchestrate.log
"""
import sys, os, subprocess, time, json, argparse, threading
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
LOGFILE = BASE / "orchestrate.log"
MODAL_DIR = str(BASE)

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

# ── epoch inspector (reuses torch-only image, fast startup) ─────────────────
_check_image = None

def get_epoch(ckpt_name: str) -> dict:
    """Return {'epoch': N, 'best_f1': X} from the resume checkpoint on the volume."""
    import modal
    global _check_image
    if _check_image is None:
        _check_image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")
    volume = modal.Volume.from_name("bionne-v2")
    tmp_app = modal.App("check-epoch-tmp")

    @tmp_app.function(image=_check_image, volumes={"/vol": volume}, timeout=120, cpu=2)
    def _check(name):
        import torch
        from pathlib import Path
        volume.reload()
        p = Path("/vol/checkpoints") / (name + ".resume")
        if not p.exists():
            return {"epoch": 0, "best_f1": 0.0}
        s = torch.load(str(p), map_location="cpu", weights_only=False)
        return {"epoch": s.get("epoch", 0),
                "best_f1": s.get("best_macro_f1", s.get("best_f1", 0.0))}

    with tmp_app.run():
        return _check.remote(ckpt_name)


def run_cmd(cmd: list, label: str) -> int:
    """Run a modal command, stream output to log, return exit code."""
    log(f"[{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=MODAL_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        with open(LOGFILE, "a") as f:
            f.write(f"  {line}\n")
        if any(k in line for k in ("Epoch", "epoch", "F1", "loss", "Error", "error", "Saving", "Resuming")):
            print(f"  [{label}] {line}", flush=True)
    proc.wait()
    return proc.returncode


def wait_pid(pid: int, label: str):
    if not pid:
        return
    log(f"[{label}] Waiting for in-flight process PID {pid}...")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(10)
    log(f"[{label}] PID {pid} finished.")


def train_session(ckpt: str, train_data: str, val_data: str, label: str):
    cmd = [
        "modal", "run", "BioNNE-R/modal_app.py::train_model",
        "--train-data", train_data, "--val-data", val_data,
        "--ckpt-name", ckpt, "--model-name", "microsoft/mdeberta-v3-base",
        "--epochs", "10", "--batch-size", "128", "--use-fp16",
        "--gradient-checkpointing",
    ]
    rc = run_cmd(cmd, label)
    log(f"[{label}] Training session exit code: {rc}")


def predict_and_download(ckpt: str, data: str, out: str, label: str):
    cmd = [
        "modal", "run", "BioNNE-R/modal_app.py::predict_model",
        "--data-name", data, "--ckpt-name", ckpt, "--out-name", out,
    ]
    rc = run_cmd(cmd, label)
    if rc != 0:
        log(f"[{label}] ERROR: prediction failed (exit {rc})")
        return
    # Download
    dest = str(BASE / "baseline" / "outputs" / out)
    dl_cmd = ["modal", "volume", "get", "bionne-v2", f"predictions/{out}", dest]
    rc2 = run_cmd(dl_cmd, label)
    if rc2 == 0:
        log(f"[{label}] Saved to {dest}")
    else:
        log(f"[{label}] WARNING: download failed (exit {rc2})")


def run_track(ckpt: str, train_data: str, val_data: str,
              test_data: str, out: str, label: str, wait_pid_val: int):
    wait_pid(wait_pid_val, label)

    while True:
        log(f"[{label}] Checking epoch...")
        try:
            info = get_epoch(ckpt)
        except Exception as e:
            log(f"[{label}] epoch check failed: {e}. Retrying in 60s...")
            time.sleep(60)
            continue

        epoch = info["epoch"]
        best_f1 = info["best_f1"]
        log(f"[{label}] epoch={epoch}/10  best_f1={best_f1:.4f}")

        if epoch >= 10:
            log(f"[{label}] Training complete — running prediction.")
            predict_and_download(ckpt, test_data, out, label)
            return

        log(f"[{label}] Resuming from epoch {epoch} → 10...")
        train_session(ckpt, train_data, val_data, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ru-pid",  type=int, default=0)
    parser.add_argument("--bil-pid", type=int, default=0)
    parser.add_argument("--en-pid",  type=int, default=0)
    args = parser.parse_args()

    log("=== Orchestration started ===")

    tracks = [
        dict(ckpt="mdeberta_ru_v2.pt",        train_data="rus_train.txt",
             val_data="rus_dev.txt",            test_data="rus_test_blind.txt",
             out="rus_test_pred.tsv",           label="RU",
             wait_pid_val=args.ru_pid),
        dict(ckpt="mdeberta_bilingual_v2.pt",  train_data="bilingual_train.txt",
             val_data="bilingual_dev.txt",      test_data="bilingual_test_blind.txt",
             out="bilingual_test_pred.tsv",     label="BIL",
             wait_pid_val=args.bil_pid),
        dict(ckpt="mdeberta_en_v1.pt",         train_data="eng_train_aug.txt",
             val_data="eng_dev.txt",            test_data="eng_test_blind.txt",
             out="eng_test_pred_v2.tsv",        label="EN",
             wait_pid_val=args.en_pid),
    ]

    threads = [threading.Thread(target=run_track, kwargs=t, daemon=False)
               for t in tracks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log("=== All done ===")
    outputs = list((BASE / "baseline" / "outputs").iterdir())
    log("Outputs: " + ", ".join(p.name for p in outputs))


if __name__ == "__main__":
    main()
