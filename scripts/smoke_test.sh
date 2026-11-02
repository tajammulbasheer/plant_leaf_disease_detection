#!/usr/bin/env bash
# End-to-end smoke test: builds a tiny synthetic PlantVillage / PlantDoc layout OUTSIDE the repo
# and runs every pipeline script with minimal settings. Checks that the code runs and writes its
# outputs; the numbers it produces are meaningless and must never be reported.
#
#   scripts/smoke_test.sh                 # fp32, temp dir (deleted on success)
#   AMP=1 scripts/smoke_test.sh           # also exercise --amp
#   SMOKE_DIR=/tmp/smoke KEEP=1 scripts/smoke_test.sh
#
# Env: SMOKE_DIR (work dir, default: mktemp), KEEP=1 (keep it), AMP=1, NUM_WORKERS (default 2),
#      PYTHON (default: .venv/bin/python if present), ALLOW_CPU=1 (don't require a GPU; slow).
set -Eeuo pipefail  # -E: ERR trap also fires inside functions

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d -t plant_smoke.XXXXXX)}"
NW="${NUM_WORKERS:-2}"
AMP_FLAG=""; [ "${AMP:-0}" = "1" ] && AMP_FLAG="--amp"
# Flags added for local runs; only passed when the scripts support them (checked via --help).
supports() { "$PY" -m "$1" --help 2>/dev/null | grep -q -- "$2"; }
opt() { local mod=$1; shift; local out=""; for f in "$@"; do supports "$mod" "${f%% *}" && out="$out $f"; done; echo "$out"; }

# The real README.md and assets/ must come out untouched; compare checksums before vs after.
guard_sum() { find README.md assets -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum; }
GUARD="$(guard_sum)"
case "$SMOKE_DIR" in "$REPO"*) echo "SMOKE_DIR must be outside the repo"; exit 1;; esac
STEP=""
trap 'echo; echo "SMOKE TEST FAILED at step: ${STEP}  (work dir: $SMOKE_DIR)"' ERR
step() { STEP="$1"; echo; echo "=== [$(date +%T)] $1"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "missing expected output: $f"; false; }; done; }

W="$SMOKE_DIR/work"; D="$SMOKE_DIR/data"; R="$SMOKE_DIR/runs"; PD="$SMOKE_DIR/PlantDoc-Dataset/test"
echo "python: $PY | work dir: $SMOKE_DIR | AMP=${AMP:-0} | workers=$NW"
if [ "${ALLOW_CPU:-0}" = "1" ]; then
    "$PY" -c "import torch; print(torch.__version__, 'CUDA' if torch.cuda.is_available() else 'CPU only')"
else
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available (ALLOW_CPU=1 to run on CPU)'; print(torch.__version__, torch.cuda.get_device_name(0))"
fi

step "build synthetic dataset"
"$PY" - "$W/PlantVillage-Dataset/raw" "$PD" <<'EOF'
import sys, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
raw, pd = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(0)
classes = {"Apple___Apple_scab": (90, 120, 40), "Apple___Black_rot": (60, 40, 30),
           "Apple___Cedar_apple_rust": (200, 120, 30), "Apple___healthy": (40, 160, 50),
           "Tomato___healthy": (30, 140, 90)}
def leaf(color, size=128):
    img = Image.new("RGB", (size, size), (200, 200, 200))
    d = ImageDraw.Draw(img)
    box = [rng.randint(5, 30), rng.randint(5, 30), rng.randint(95, 122), rng.randint(95, 122)]
    jitter = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in color)
    d.ellipse(box, fill=jitter)
    return img, box
for name, color in classes.items():
    (raw / "color" / name).mkdir(parents=True, exist_ok=True)
    (raw / "segmented" / name).mkdir(parents=True, exist_ok=True)
    for i in range(40):
        img, box = leaf(color)
        stem = f"{i:04d}_{name[:6]}"
        img.save(raw / "color" / name / f"{stem}.JPG")
        seg = Image.new("RGB", img.size, (0, 0, 0))
        ImageDraw.Draw(seg).ellipse(box, fill=(120, 180, 80))
        seg.save(raw / "segmented" / name / f"{stem}_final_masked.jpg")
pd_map = {"Apple Scab Leaf": "Apple___Apple_scab", "Apple rust leaf": "Apple___Cedar_apple_rust",
          "Apple leaf": "Apple___healthy", "Tomato leaf": "Tomato___healthy"}
for folder, cls in pd_map.items():
    (pd / folder).mkdir(parents=True, exist_ok=True)
    for i in range(6):
        leaf(classes[cls])[0].save(pd / folder / f"{i}.jpg")
print("synthetic data written")
EOF

step "prepare_plantvillage"
"$PY" -m src.data.prepare_plantvillage --output-dir "$D" --workdir "$W"
need "$D/plantvillage_full/train" "$D/apple/val" "$D/seg/images" "$D/seg/masks"
[ "$(ls "$D/plantvillage_full/train" | wc -l)" = 5 ] && [ "$(ls "$D/apple/train" | wc -l)" = 4 ]

CLS_OPTS="--batch-size 8 --image-size 64 $(opt src.training.train_classifier "--num-workers $NW" "$AMP_FLAG")"

step "train_classifier --mode full"
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$R/baseline_full" \
    --mode full --epochs 2 $CLS_OPTS
need "$R/baseline_full/history.json" "$R/baseline_full/best_model.pt" "$R/baseline_full/classes.json"

step "train_classifier --resume (extend 2 -> 3 epochs, then no-op)"
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$R/baseline_full" \
    --mode full --epochs 3 --resume $CLS_OPTS | tee "$SMOKE_DIR/resume1.txt"
grep -q "Resuming from last_checkpoint.pt: 2 epoch" "$SMOKE_DIR/resume1.txt"
[ "$("$PY" -c "import json;print(len(json.load(open('$R/baseline_full/history.json'))))")" = 3 ]
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$R/baseline_full" \
    --mode full --epochs 3 --resume $CLS_OPTS | tee "$SMOKE_DIR/resume2.txt"
grep -q "Run already complete" "$SMOKE_DIR/resume2.txt"

step "train_classifier --resume on a run without last_checkpoint.pt (runs made before it existed)"
cp -r "$R/baseline_full" "$SMOKE_DIR/legacy_run" && rm "$SMOKE_DIR/legacy_run/last_checkpoint.pt"
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$SMOKE_DIR/legacy_run" \
    --mode full --epochs 4 --resume $CLS_OPTS | tee "$SMOKE_DIR/resume3.txt"
grep -q "no last_checkpoint.pt: continuing from best-epoch weights" "$SMOKE_DIR/resume3.txt"
need "$SMOKE_DIR/legacy_run/last_checkpoint.pt"

step "train_classifier --mode few_shot"
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$R/few_shot_k20" \
    --mode few_shot --k-shot 5 --epochs 2 $CLS_OPTS
need "$R/few_shot_k20/history.json"

step "train_classifier --mode semi_supervised (threshold 0 so pseudo-labels are always used)"
"$PY" -m src.training.train_classifier --data-dir "$D/plantvillage_full" --output-dir "$R/semi_supervised_k20" \
    --mode semi_supervised --k-shot 5 --epochs 2 --warmup-epochs 1 --confidence-threshold 0.0 \
    --max-unlabeled-per-epoch 40 $CLS_OPTS | tee "$SMOKE_DIR/semi.txt"
grep -q "pseudo-labels: 40/40" "$SMOKE_DIR/semi.txt"

step "train_classifier Apple-only"
"$PY" -m src.training.train_classifier --data-dir "$D/apple" --output-dir "$R/baseline_apple_only" \
    --mode full --epochs 2 $CLS_OPTS
need "$R/baseline_apple_only/best_model.pt"

step "train_segmentation"
"$PY" -m src.training.train_segmentation --images-dir "$D/seg/images" --masks-dir "$D/seg/masks" \
    --output-dir "$R/unet" --epochs 2 --batch-size 4 --image-size 128 \
    $(opt src.training.train_segmentation "--num-workers $NW" "$AMP_FLAG")
need "$R/unet/seg_results.json" "$R/unet/seg_samples.png" "$R/unet/unet_best.pt"

step "train_gan"
"$PY" -m src.training.train_gan --data-dir "$D/apple/train/Apple___Cedar_apple_rust" \
    --output-dir "$R/gan_cedar_rust" --epochs 2 --batch-size 8 --sample-every 1 \
    $(opt src.training.train_gan "--num-workers $NW")
need "$R/gan_cedar_rust/real_vs_generated.png" "$R/gan_cedar_rust/generator.pt"

step "cross_dataset_eval"
"$PY" -m src.eval.cross_dataset_eval --checkpoint "$R/baseline_apple_only/best_model.pt" \
    --plantvillage-val-dir "$D/apple/val" --plantdoc-dir "$PD" --output-dir "$R/cross_dataset_eval" \
    --image-size 64 --batch-size 8 $(opt src.eval.cross_dataset_eval "--num-workers $NW")
need "$R/cross_dataset_eval/cross_dataset_results.json" "$R/cross_dataset_eval/confusion_plantdoc.png"

step "cross_dataset_eval rejects the 38-class checkpoint with a clear error"
if "$PY" -m src.eval.cross_dataset_eval --checkpoint "$R/baseline_full/best_model.pt" \
    --plantvillage-val-dir "$D/apple/val" --plantdoc-dir "$PD" --output-dir "$SMOKE_DIR/should_fail" \
    > "$SMOKE_DIR/wrong_ckpt.txt" 2>&1; then echo "expected failure"; false; fi
grep -q "needs the 4 Apple classes" "$SMOKE_DIR/wrong_ckpt.txt"

step "pseudo_boxes (Grad-CAM)"
"$PY" -m src.interpretability.pseudo_boxes --checkpoint "$R/baseline_apple_only/best_model.pt" \
    --data-dir "$D/apple/val" --output-dir "$R/gradcam_boxes" --num-samples 4 --image-size 64
[ "$(ls "$R/gradcam_boxes"/*.png | wc -l)" = 4 ]

step "train_incremental"
"$PY" -m src.incremental.train_incremental --data-dir "$D/apple" --output-dir "$R/incremental" \
    --k-shot 5 --epochs-per-task 1 --exemplars-per-class 3 --image-size 64 --batch-size 8 \
    $(opt src.incremental.train_incremental "--num-workers $NW" "$AMP_FLAG")
need "$R/incremental/incremental_results.json"

step "generate_report (README copy, never the real README)"
cp README.md "$SMOKE_DIR/README.copy.md"
"$PY" -m src.utils.generate_report --run-dir "$R" --output-dir "$R/report" \
    --assets-dir "$SMOKE_DIR/assets" --readme "$SMOKE_DIR/README.copy.md"
need "$R/report/RESULTS.md" "$SMOKE_DIR/assets/classifier_modes.png" "$SMOKE_DIR/assets/incremental_matrix.png"
grep -q "RESULTS:START" "$SMOKE_DIR/README.copy.md"
if grep -q "not run yet" "$SMOKE_DIR/README.copy.md"; then echo "report still has not-run sections"; false; fi
[ "$(guard_sum)" = "$GUARD" ] || { echo "smoke test modified the real README.md or assets/"; false; }

trap - ERR
echo; echo "SMOKE TEST PASSED (AMP=${AMP:-0})"
if [ "${KEEP:-0}" = "1" ]; then echo "kept: $SMOKE_DIR"; else rm -rf "$SMOKE_DIR"; fi
