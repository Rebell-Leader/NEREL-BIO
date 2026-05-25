#!/usr/bin/env bash
# Auto-resume RU and bilingual mDeBERTa training, then run predictions.
# Usage: bash orchestrate.sh [--ru-pid PID] [--bil-pid PID]
#   If PIDs are passed, wait for those processes first before checking epoch.
# Logs to: orchestrate.log

set -euo pipefail
LOGFILE="$(dirname "$0")/orchestrate.log"
MODAL_DIR="$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

RU_PID=""
BIL_PID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ru-pid)  RU_PID="$2";  shift 2 ;;
        --bil-pid) BIL_PID="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Wait for a PID if set (training session that was already running)
wait_pid() {
    local pid="$1" label="$2"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Waiting for $label (PID $pid) to finish..."
        wait "$pid" || true
        log "$label process exited."
    fi
}

# Get current epoch for a checkpoint from the volume
get_epoch() {
    local ckpt="$1"
    python3 - "$ckpt" <<'PYEOF'
import sys, subprocess, json
ckpt = sys.argv[1]
result = subprocess.run(
    ["modal", "run", "BioNNE-R/modal_app.py::inspect_checkpoints",
     "--ckpt-names", f'["{ckpt}"]'],
    capture_output=True, text=True, cwd="/home/obi/competitions/NEREL-BIO"
)
# extract JSON dict from output — it's on the last non-empty line
for line in reversed(result.stdout.splitlines()):
    line = line.strip()
    if line.startswith('{'):
        try:
            data = json.loads(line)
            info = data.get(ckpt, {})
            print(f"epoch={info.get('epoch',0)} best_f1={info.get('best_f1',0):.4f}")
            sys.exit(0)
        except: pass
print("epoch=0 best_f1=0.0000")
PYEOF
}

# Run one training session and wait for it to finish
run_train() {
    local ckpt="$1" train_data="$2" val_data="$3" label="$4"
    log "[$label] Starting training session (resume from epoch in $ckpt)..."
    modal run BioNNE-R/modal_app.py::train_model \
        --train-data "$train_data" --val-data "$val_data" \
        --ckpt-name "$ckpt" --model-name microsoft/mdeberta-v3-base \
        --epochs 10 --batch-size 128 --use-fp16 --gradient-checkpointing \
        2>&1 | tee -a "$LOGFILE"
    log "[$label] Training session finished."
}

# Run prediction and download result
run_predict() {
    local ckpt="$1" data="$2" out="$3" label="$4"
    log "[$label] Running prediction: $data → $out"
    modal run BioNNE-R/modal_app.py::predict_model \
        --data-name "$data" --ckpt-name "$ckpt" --out-name "$out" \
        2>&1 | tee -a "$LOGFILE"
    log "[$label] Prediction done. Downloading $out..."
    modal volume get bionne-v2 "predictions/$out" \
        "$(dirname "$0")/baseline/outputs/$out" \
        2>&1 | tee -a "$LOGFILE"
    log "[$label] Saved to baseline/outputs/$out"
}

# ── Track function: loop until epoch==10, then predict ──────────────────────
run_track() {
    local ckpt="$1" train_data="$2" val_data="$3" test_data="$4" out="$5" label="$6" wait_pid_val="$7"

    # Wait for in-flight session if one was already running
    wait_pid "$wait_pid_val" "$label"

    while true; do
        log "[$label] Checking epoch..."
        epoch_info=$(get_epoch "$ckpt")
        epoch=$(echo "$epoch_info" | grep -oP 'epoch=\K[0-9]+')
        best_f1=$(echo "$epoch_info" | grep -oP 'best_f1=\K[0-9.]+')
        log "[$label] $epoch_info"

        if [[ "$epoch" -ge 10 ]]; then
            log "[$label] Training complete (epoch $epoch/10, best_f1=$best_f1). Running prediction."
            run_predict "$ckpt" "$test_data" "$out" "$label"
            break
        fi

        log "[$label] Epoch $epoch/10 — resuming training..."
        run_train "$ckpt" "$train_data" "$val_data" "$label"
    done
}

log "=== Orchestration started ==="

# Run both tracks in parallel
run_track \
    mdeberta_ru_v2.pt rus_train.txt rus_dev.txt \
    rus_test_blind.txt rus_test_pred.tsv "RU" "$RU_PID" &
RU_ORCH_PID=$!

run_track \
    mdeberta_bilingual_v2.pt bilingual_train.txt bilingual_dev.txt \
    bilingual_test_blind.txt bilingual_test_pred.tsv "BIL" "$BIL_PID" &
BIL_ORCH_PID=$!

log "RU orchestration PID: $RU_ORCH_PID"
log "BIL orchestration PID: $BIL_ORCH_PID"

wait "$RU_ORCH_PID" || log "RU track exited with error"
wait "$BIL_ORCH_PID" || log "BIL track exited with error"

log "=== All done. ==="
log "Predictions in baseline/outputs/:"
ls "$(dirname "$0")/baseline/outputs/" | tee -a "$LOGFILE"
