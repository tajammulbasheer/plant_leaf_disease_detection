# Running locally (WSL2 / Linux, one CUDA GPU)

`scripts/run_all.sh` runs the same steps, in the same order and with the same settings, as
`notebooks/train_on_colab.ipynb`. Local-only options (`AMP`, `NUM_WORKERS`, `BATCH_SIZE`) are opt-in;
with none of them set, every script gets exactly the arguments the notebook gives it.

## 1. Setup

Keep the repo and the data on the Linux filesystem (`~/...`), not under `/mnt/c`: reading tens of
thousands of small JPEGs through `/mnt/c` is many times slower. Run everything from a WSL shell, not
through `\\wsl.localhost` from Windows.

```bash
cd ~/projects/plant_leaf_disease_detection
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# pick the wheel matching your driver's CUDA version (nvidia-smi, top right); cu130 = CUDA 13.0
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
scripts/smoke_test.sh            # every script on tiny synthetic data
```

Disk: about 10 GB for data and runs (the PlantVillage clone is deleted after preparation).

## 2. Run the pipeline

```bash
scripts/run_all.sh
```

| Variable | Default | Effect |
|---|---|---|
| `DATA_DIR` | `~/data/plantvillage` | prepared data (`plantvillage_full/`, `apple/`, `seg/`) and the PlantDoc clone |
| `RUN_DIR` | `~/runs/plant_leaf` | one folder per experiment, plus `logs/` and `report/` |
| `AMP` | `0` | `1` passes `--amp` (fp16 mixed precision) to the classifier, U-Net and incremental steps |
| `BATCH_SIZE` | script default (32) | batch size of the classifier steps and the PlantDoc eval |
| `NUM_WORKERS` | `8` | DataLoader workers for every step |
| `PYTHON` | `.venv/bin/python` | interpreter |

Steps: data preparation → 38-class classifier (full, few-shot, semi-supervised) → Apple-only classifier
→ U-Net → DCGAN → PlantDoc eval → Grad-CAM boxes → class-incremental → `generate_report`, which writes
the figures into `assets/` and fills the results block of `README.md`.

Each step logs to `$RUN_DIR/logs/<step>.log`, starting with the date, git commit, torch / CUDA / cuDNN /
driver versions and the exact command. Before every training step the free GPU memory is printed, with a
warning below 6 GB. If a step fails, the script stops and prints its name and log file.

Note: `AMP=1` and a non-default `BATCH_SIZE` change how the models are trained, so results are not
strictly comparable with the Colab (fp32, default batch size) runs. `generate_report` records which runs
used AMP under the results tables.

## 3. Speed on an RTX A4000 (measured 2026-11-02)

`scripts/benchmark_amp.py` times real training steps on the prepared data (30 warm-up batches, then 100
timed, default batch sizes, torch 2.14.1 + CUDA 13.0). **The GPU was shared** with another training/eval
job during every measurement, so throughput varied a lot between repeats (4 runs):

| | fp32 | AMP | Peak GPU memory fp32 → AMP |
|---|---|---|---|
| Classifier (EfficientNet-B0, 224 px, batch 32) | 190–345 img/s | 298–434 img/s | 2839 → 1465 MiB (×0.52) |
| U-Net (256 px, batch 8) | 45–61 img/s | 75–180 img/s | 1554 → 889 MiB (×0.57) |

The memory saving is exact and repeatable. The speed-up is **not reliably measured**: the spread between
repeats is larger than the AMP effect. Re-measure on an idle GPU (`nvidia-smi` shows no other process):

```bash
.venv/bin/python scripts/benchmark_amp.py --data-dir ~/data/plantvillage            # GPU step only
.venv/bin/python scripts/benchmark_amp.py --data-dir ~/data/plantvillage --mode loader  # incl. JPEG loading
```

**Estimated** time per step, fp32, derived from the measured fp32 range above and the dataset sizes
(43,429 / 10,876 train / val images; Apple 2,536 / 635; U-Net 2,854 / 317). Validation is assumed to run
~3× faster than training (not measured). The DCGAN, Grad-CAM and PlantDoc steps were not benchmarked.
These are estimates, not measurements:

| Step | Work | Estimate |
|---|---|---|
| Data preparation | clone ~2 GB + copy 54k images | ~30 min here (download-bound, ~7 MB/s) |
| Classifier, full | 20 epochs × 43k train + 11k val | 45–85 min |
| Classifier, few-shot | 25 epochs × 760 train + 11k val | 5–10 min |
| Classifier, semi-supervised | as few-shot + up to 2,000 pseudo-labels / epoch | 10–15 min |
| Classifier, Apple-only | 15 epochs × 2.5k train | 2–4 min |
| U-Net | 15 epochs × 2.9k images | 12–16 min |
| DCGAN | 100 epochs × 220 images at 64 px | a few minutes (not measured) |
| PlantDoc eval, Grad-CAM | inference on < 700 images | ~1 min each |
| Class-incremental | 2 strategies × 3 tasks × 15 epochs (batch 16) | 5–10 min |
| **Total** (data already prepared) | | **~1.5–2.5 h** |

On an idle GPU expect the lower end or better; with another job on the card, the upper end or worse.

## 4. Long runs in tmux

```bash
sudo apt install -y tmux                       # once, if missing
tmux new -s plant                              # start a session
cd ~/projects/plant_leaf_disease_detection && scripts/run_all.sh   # each step also logs to ~/runs/plant_leaf/logs/
# detach: Ctrl-b then d          reattach later: tmux attach -t plant
```

Watch progress from another terminal:

```bash
ls -t ~/runs/plant_leaf/logs/                                  # newest log = current step
tail -f "$(ls -t ~/runs/plant_leaf/logs/*.log | head -1)"
watch -n 2 nvidia-smi
```

**tmux does not survive** Windows sleep or hibernation, a Windows reboot or update restart, or
`wsl --shutdown`. All of these stop the WSL VM and kill the run. To avoid them:

- Windows Settings → System → Power: set **Sleep** to *Never* while plugged in (and lid close to
  *Do nothing* on a laptop).
- Windows Update → **Pause updates** for the duration, or set *Active hours* to cover the run.
- Don't run `wsl --shutdown` or close the distro from Docker Desktop / VS Code while training.
- WSL also shuts the VM down a while after the last WSL terminal closes. Keep one WSL terminal window
  open (the tmux session inside it is enough), or on recent WSL set `vmIdleTimeout=-1` under `[wsl2]`
  in `%UserProfile%\.wslconfig`.

## 5. Resuming

Just run `scripts/run_all.sh` again (in tmux). Nothing needs to be deleted.

| Step | After an interruption |
|---|---|
| Data preparation | Skipped once `plantvillage_full/val` and `seg/masks` exist. An interrupted clone is removed and re-cloned. |
| Classifiers (4 runs) | **Resume** with `--resume` from `last_checkpoint.pt` (weights, optimizer, LR schedule, RNG state of the last finished epoch). A finished run prints "Run already complete". Loses at most the epoch in progress. |
| U-Net | Restarts from scratch (skipped once `seg_samples.png` exists). |
| DCGAN | Restarts from scratch (skipped once `real_vs_generated.png` exists). |
| PlantDoc eval | Re-runs (skipped once `cross_dataset_results.json` exists). |
| Grad-CAM | Re-runs (skipped once all 16 images exist). |
| Class-incremental | Restarts both strategies from scratch (skipped once `incremental_results.json` exists). |
| Report | Always re-runs (seconds). |

To redo a step, delete its folder under `$RUN_DIR`. Classifier runs from before `last_checkpoint.pt`
existed still resume, but from the best epoch's weights with a fresh optimizer (printed in the log).

## 6. Results into the repo

`run_all.sh` ends with `generate_report --readme README.md --assets-dir assets`, so the README results
block and `assets/*.png` are updated in place. Review and commit them:

```bash
git diff README.md
git status assets/
```

Never edit numbers in the results block by hand. Weights, runs and logs stay out of git (`.gitignore`).

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` | Another process holds GPU memory (check `nvidia-smi`), or the batch is too large. Free the GPU, or rerun with `AMP=1` (roughly halves activation memory), or `BATCH_SIZE=16`. Classifier steps resume where they stopped. |
| Training is far slower than expected, GPU-Util at 100% before you started | **Shared GPU**: another job is computing on the same card. `nvidia-smi` lists it. Both jobs slow down; WSL may also spill GPU memory into system RAM when the card is full, which is much slower still. Wait for it to finish or stop it. |
| Process `Killed` with no Python traceback, or WSL freezes | **WSL memory limit** reached (Linux OOM killer). Check `free -h`; lower `NUM_WORKERS` (e.g. 4); raise the limit with `memory=24GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown` (not during a run) to apply. |
| Progress bar stops, CPU idle, no error (**DataLoader hang**) | Usually a worker died or shared memory ran out. Ctrl-C, rerun with `NUM_WORKERS=0` to see the real error; if it then works, use fewer workers. Check `df -h /dev/shm`. |
| `trained on 38 classes, but this script needs the 4 Apple classes` | The Apple classifier step hasn't finished; PlantDoc eval and Grad-CAM need `baseline_apple_only/best_model.pt`. |
| `torch.cuda.is_available()` is False | Install the Windows NVIDIA driver (not a Linux driver inside WSL), and a torch wheel whose CUDA version the driver supports. |
| Slow data loading | Repo or data under `/mnt/c`; move to `~`. |
