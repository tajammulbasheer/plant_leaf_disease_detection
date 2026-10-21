# Running on Google Colab

Everything runs from `notebooks/train_on_colab.ipynb`. The notebook is only a launcher: each cell
calls one script in `src/`. Nothing needs uploading: the notebook clones this repo itself. No Kaggle
or Hugging Face account is needed.

**Open it directly:** [Open in Colab](https://colab.research.google.com/github/tajammulbasheer/plant_leaf_disease_detection/blob/main/notebooks/train_on_colab.ipynb)

## Where things live

| What | Where | Survives a disconnect? |
|---|---|---|
| Code | `/content/plant_leaf_disease_detection` (cloned from GitHub by cell 2) | No, cell 2 re-clones it each session |
| Data | `/content/data` (built by cell 4) | No, re-run cell 4 each session |
| Results, checkpoints, figures | Google Drive: `MyDrive/apple_leaf_runs/` | **Yes** |

## First run

1. Open the notebook from GitHub: [this link](https://colab.research.google.com/github/tajammulbasheer/plant_leaf_disease_detection/blob/main/notebooks/train_on_colab.ipynb), or in Colab use
   **File → Open notebook → GitHub**, search `tajammulbasheer/plant_leaf_disease_detection`.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run cells in order:

| Cell | Step | Output folder in Drive | Approx. time on T4 |
|---|---|---|---|
| 1 | GPU check | — | seconds |
| 2 | Clone (or update) the repo from GitHub, install | — | 1–2 min |
| 3 | Mount Drive | — | seconds |
| 4 | Prepare data (clone, split, Apple subset, masks) | — (local disk) | 10–15 min |
| 5a | 38-class classifier, full | `baseline_full/` | longest step, ~1.5–3 h |
| 5b | 38-class classifier, few-shot | `few_shot_k20/` | ~30–45 min |
| 5c | 38-class classifier, semi-supervised | `semi_supervised_k20/` | ~45–75 min |
| 6 | 4-class Apple classifier | `baseline_apple_only/` | ~10 min |
| 7 | U-Net segmentation | `unet/` | ~15–25 min |
| 8 | DCGAN (optional) | `gan_cedar_rust/` | ~20–30 min |
| 9 | PlantDoc cross-dataset test (needs cell 6) | `cross_dataset_eval/` | ~2 min |
| 10 | Grad-CAM boxes (needs cell 6) | `gradcam_boxes/` | ~1 min |
| 11 | Class-incremental learning | `incremental/` | ~20–30 min |
| 12 | Build report + fill README | `report/` | seconds |
| 13 | Download `report.zip` | — | seconds |

Times are estimates and vary with Colab load.

## After a disconnect (or a new day)

Re-run cells **1 → 4**, then carry on. You don't need to work out where you stopped:

- Classifier cells (5a–5c, 6) use `--resume`: they reload the last saved checkpoint and continue from the
  next epoch. A finished run prints "Run already complete" and exits.
- Other cells print "Already done, skipping" when their result file is already in Drive.

To redo a step from scratch, delete its folder in `MyDrive/apple_leaf_runs/` first.

## Updating from an older version of this repo

If you ran an earlier version, these results were produced by code that has since been fixed. Delete
these folders in Drive and re-run the matching cells:

| Delete | Why | Re-run cell |
|---|---|---|
| `semi_supervised_k20/` | pseudo-labeling used to load every selected image into memory and train on them in one batch (this is what killed the process) | 5c |
| `incremental/` | evaluation could predict classes not yet learned; replay buffer wasn't refreshed between tasks | 11 |
| `gan_cedar_rust/` (if present) | GAN was trained on all Apple classes instead of one | 8 |

`baseline_full/`, `few_shot_k20/`, `baseline_apple_only/` and `cross_dataset_eval/` are unaffected; keep them.

## Avoiding disconnects

- Keep the Colab tab open and your computer awake. A sleeping laptop or a closed lid ends the session.
- Free Colab has daily GPU limits. If you hit one, wait or switch to Colab Pro; resume picks up where it left off.
- If output just stops with `^C` and no error, the session was killed (time limit, idle, or memory).
  Re-run cells 1–4 and the same cell again.

## Getting results into the repo

Cell 13 downloads `report.zip`, containing:

```
report/
  README.md      the project README with the results block filled in
  RESULTS.md     the same tables on their own
  assets/*.png   all charts and sample images
```

Copy `README.md` over the repo's README and the PNGs into the repo's `assets/` folder, then commit
and push. The README already references images as `assets/<name>.png`.

Code changes pushed to GitHub reach Colab automatically: cell 2 runs `git pull` each session.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No such file ... /content/data/...` | New session: run cell 4 again |
| `trained on 38 classes, but this script needs the 4 Apple classes` | Run cell 6 first; cells 9 and 10 use the Apple checkpoint |
| `CUDA out of memory` | Add `--batch-size 16` to that cell |
| Clone in cell 4 fails partway | Re-run cell 4; it retries cleanly |
| `RESULTS:START markers` warning in cell 12 | The README was edited and lost the marker comments; restore them from the repo |
