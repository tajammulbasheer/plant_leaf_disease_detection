#!/usr/bin/env bash
# Full local pipeline: same steps, order and settings as notebooks/train_on_colab.ipynb.
# Safe to re-run after an interruption: finished steps are skipped and classifier runs --resume.
#
#   scripts/run_all.sh
#   AMP=1 BATCH_SIZE=64 scripts/run_all.sh
#
# Env vars (all optional):
#   DATA_DIR      prepared data + PlantDoc clone      (default ~/data/plantvillage)
#   RUN_DIR       checkpoints, results, logs          (default ~/runs/plant_leaf)
#   AMP=1         pass --amp to the classifier, U-Net and incremental steps (default 0 = fp32, as on Colab)
#   BATCH_SIZE    override --batch-size for the classifier steps and the PlantDoc eval
#                 (default: unset = each script's own default, as on Colab)
#   NUM_WORKERS   DataLoader workers for every step (default 8)
#   PYTHON        interpreter (default .venv/bin/python)
set -Eeuo pipefail  # -E: ERR trap also fires inside functions

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
DATA="${DATA_DIR:-$HOME/data/plantvillage}"
RUN="${RUN_DIR:-$HOME/runs/plant_leaf}"
LOGS="$RUN/logs"
NW="${NUM_WORKERS:-8}"
AMP_FLAG=""; [ "${AMP:-0}" = "1" ] && AMP_FLAG="--amp"
BS_FLAG=""; [ -n "${BATCH_SIZE:-}" ] && BS_FLAG="--batch-size $BATCH_SIZE"
PLANTDOC="$DATA/PlantDoc-Dataset"
MIN_FREE_MB=6144
export PYTHONUNBUFFERED=1

mkdir -p "$DATA" "$RUN" "$LOGS"
STEP="setup"
trap 'echo; echo "*** run_all.sh FAILED at step: $STEP  (log: $LOGS/$STEP.log)"; echo "*** Fix the cause and re-run; finished steps will be skipped."' ERR

versions() {
    echo "date:     $(date -Is)"
    echo "git:      $(git rev-parse HEAD)$(git diff --quiet HEAD -- src scripts || echo ' (uncommitted changes in src/ or scripts/)')"
    "$PY" -c "import torch, torchvision; print(f'torch:    {torch.__version__} | torchvision {torchvision.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}'); print(f'GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE (CPU only)\"}')"
    echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
    echo "settings: AMP=${AMP:-0} BATCH_SIZE=${BATCH_SIZE:-default} NUM_WORKERS=$NW DATA_DIR=$DATA RUN_DIR=$RUN"
}

gpu_check() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
    if [ -z "$free" ]; then echo "WARNING: nvidia-smi not available, cannot check GPU memory"; return; fi
    echo "GPU free memory: ${free} MiB"
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "WARNING: less than $((MIN_FREE_MB / 1024)) GB of GPU memory free -- another process is using the GPU."
        echo "         This step may run out of memory or run very slowly. Check: nvidia-smi"
    fi
}

# run <step-name> <is-training 0|1> <command...>: header + versions in the log, output to console and log.
run() {
    STEP="$1"; local training="$2"; shift 2
    local log="$LOGS/$STEP.log"
    echo; echo "=================== [$(date +%T)] $STEP  (log: $log)"
    { echo "=== $STEP"; versions; echo "command:  $*"; } >> "$log"
    [ "$training" = 1 ] && gpu_check | tee -a "$log"
    "$@" 2>&1 | tee -a "$log"
    echo "=== $STEP finished $(date -Is)" >> "$log"
}

skip() { echo; echo "=================== $1: already done, skipping ($2)"; }

versions | tee -a "$LOGS/run_all.log"

# 1. Data (notebook cell 4). Considered prepared once the split and the U-Net masks exist.
if [ -d "$DATA/plantvillage_full/val" ] && [ -n "$(ls -A "$DATA/seg/masks" 2>/dev/null)" ]; then
    skip prepare_data "$DATA/plantvillage_full"
else
    run prepare_data 0 "$PY" -m src.data.prepare_plantvillage --output-dir "$DATA" --workdir "$DATA" --cleanup-clone
fi

# 2. 38-class classifier, three modes (cells 5a-5c). --resume: continue or no-op when complete.
run classifier_full 1 "$PY" -m src.training.train_classifier --data-dir "$DATA/plantvillage_full" \
    --output-dir "$RUN/baseline_full" --mode full --epochs 20 --resume \
    --num-workers "$NW" $BS_FLAG $AMP_FLAG
run classifier_few_shot 1 "$PY" -m src.training.train_classifier --data-dir "$DATA/plantvillage_full" \
    --output-dir "$RUN/few_shot_k20" --mode few_shot --k-shot 20 --epochs 25 --resume \
    --num-workers "$NW" $BS_FLAG $AMP_FLAG
run classifier_semi_supervised 1 "$PY" -m src.training.train_classifier --data-dir "$DATA/plantvillage_full" \
    --output-dir "$RUN/semi_supervised_k20" --mode semi_supervised --k-shot 20 --epochs 25 \
    --confidence-threshold 0.95 --max-unlabeled-per-epoch 2000 --resume \
    --num-workers "$NW" $BS_FLAG $AMP_FLAG

# 3. Apple-only classifier (cell 6), needed by PlantDoc eval and Grad-CAM.
run classifier_apple 1 "$PY" -m src.training.train_classifier --data-dir "$DATA/apple" \
    --output-dir "$RUN/baseline_apple_only" --mode full --epochs 15 --resume \
    --num-workers "$NW" $BS_FLAG $AMP_FLAG

# 4. U-Net (cell 7). No resume: an interrupted run restarts from scratch.
if [ -f "$RUN/unet/seg_samples.png" ]; then skip unet "$RUN/unet/seg_samples.png"; else
    run unet 1 "$PY" -m src.training.train_segmentation --images-dir "$DATA/seg/images" \
        --masks-dir "$DATA/seg/masks" --output-dir "$RUN/unet" --epochs 15 \
        --num-workers "$NW" $AMP_FLAG
fi

# 5. DCGAN on the rarest Apple class (cell 8). No resume.
if [ -f "$RUN/gan_cedar_rust/real_vs_generated.png" ]; then skip gan "$RUN/gan_cedar_rust/real_vs_generated.png"; else
    run gan 1 "$PY" -m src.training.train_gan --data-dir "$DATA/apple/train/Apple___Cedar_apple_rust" \
        --output-dir "$RUN/gan_cedar_rust" --epochs 100 --num-workers "$NW"
fi

# 6. PlantDoc cross-dataset eval (cell 9).
if [ ! -d "$PLANTDOC/test" ]; then
    rm -rf "$PLANTDOC"
    run clone_plantdoc 0 git clone -q --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git "$PLANTDOC"
fi
if [ -f "$RUN/cross_dataset_eval/cross_dataset_results.json" ]; then skip plantdoc_eval "cross_dataset_results.json"; else
    run plantdoc_eval 1 "$PY" -m src.eval.cross_dataset_eval --checkpoint "$RUN/baseline_apple_only/best_model.pt" \
        --plantvillage-val-dir "$DATA/apple/val" --plantdoc-dir "$PLANTDOC/test" \
        --output-dir "$RUN/cross_dataset_eval" --num-workers "$NW" $BS_FLAG
fi

# 7. Grad-CAM pseudo boxes (cell 10). Done when all 16 images exist (the notebook only checks the
#    folder, which would also skip a run that was interrupted halfway).
if [ "$(ls "$RUN/gradcam_boxes"/*.png 2>/dev/null | wc -l)" -ge 16 ]; then skip gradcam "16 images"; else
    run gradcam 1 "$PY" -m src.interpretability.pseudo_boxes --checkpoint "$RUN/baseline_apple_only/best_model.pt" \
        --data-dir "$DATA/apple/val" --output-dir "$RUN/gradcam_boxes" --num-samples 16
fi

# 8. Class-incremental (cell 11). No resume: both strategies rerun if interrupted.
if [ -f "$RUN/incremental/incremental_results.json" ]; then skip incremental "incremental_results.json"; else
    run incremental 1 "$PY" -m src.incremental.train_incremental --data-dir "$DATA/apple" \
        --output-dir "$RUN/incremental" --k-shot 20 --epochs-per-task 15 \
        --num-workers "$NW" $AMP_FLAG
fi

# 9. Report (cell 12): RESULTS.md, figures into the repo's assets/, results block in README.md.
run report 0 "$PY" -m src.utils.generate_report --run-dir "$RUN" --output-dir "$RUN/report" \
    --assets-dir assets --readme README.md

trap - ERR
echo; echo "All steps finished. Review with: git diff README.md && git status assets/"
