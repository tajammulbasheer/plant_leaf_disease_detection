# Leaf Disease Detection: Beyond In-Distribution Accuracy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tajammulbasheer/plant_leaf_disease_detection/blob/main/notebooks/train_on_colab.ipynb)

Plant disease classification on the full **PlantVillage** dataset (38 classes, 14 crops), plus four
experiments that test what a near-perfect PlantVillage score actually means:

- **Label efficiency**: full supervision vs. 20 labels per class vs. 20 labels + pseudo-labeling
- **Generalization**: the same model tested on **PlantDoc**, real field photos of the same diseases
- **Continual learning**: adding new diseases one at a time from 20 examples each, measuring forgetting
- **Localization without boxes**: Grad-CAM heatmaps turned into lesion bounding boxes, using only image labels

Plus a U-Net leaf segmenter and a DCGAN for minority-class synthesis on the Apple subset.

## Key findings

- A standard EfficientNet-B0 reaches **99.9% accuracy** on PlantVillage, but the Apple-only version of the
  same model drops to **51.7% accuracy (0.36 macro-F1)** on PlantDoc field photos. PlantVillage's uniform
  lab backgrounds make in-distribution accuracy a poor guide to real-world performance.
- With only **20 labeled images per class**, accuracy stays at **95.1%** across all 38 classes.
- When new diseases are added sequentially, naive fine-tuning forgets almost everything it knew;
  exemplar replay + knowledge distillation substantially reduces that forgetting.

## Results

All numbers are produced by `src/utils/generate_report.py`, which fills in the block below from the
experiment outputs. Nothing here is typed by hand.

<!-- RESULTS:START -->
### Classifier: 38-class PlantVillage

| Mode | Labeled images / class | Val accuracy | Val macro-F1 | Epochs run |
|------|------------------------|--------------|--------------|------------|
| Full supervised | all | 0.9989 | 0.9986 | — |
| Few-shot | 20 | 0.9510 | 0.9383 | — |
| Semi-supervised | 20 + pseudo-labels | not run yet | — | — |

### Cross-dataset generalization (Apple classes, PlantVillage → PlantDoc)

| Test set | Accuracy | Macro-F1 |
|---|---|---|
| PlantVillage (in-distribution) | 1.0000 | 1.0000 |
| PlantDoc (field photos) | 0.5172 | 0.3606 |

Generalization gap: **0.483** accuracy, **0.639** macro-F1. *Black rot has no PlantDoc equivalent and is excluded.*

### Class-incremental few-shot learning (Apple classes)

| Strategy | Final avg. accuracy | Avg. forgetting (lower is better) |
|---|---|---|
| Naive fine-tuning | 0.3333 | 0.9066 |
| Replay + distillation | 0.4505 | 0.5965 |

### Leaf segmentation (U-Net, Apple classes)

| Val IoU | Val Dice |
|---|---|
| not run yet | — |
<!-- RESULTS:END -->

## Approach

| Component | What it does | Code |
|---|---|---|
| Data preparation | Clones the original PlantVillage repo, seeded stratified 80/20 split, Apple subset, U-Net masks | `src/data/prepare_plantvillage.py` |
| Classifier | ImageNet-pretrained EfficientNet-B0 with a small MLP head; `full`, `few_shot`, `semi_supervised` modes | `src/training/train_classifier.py` |
| Pseudo-labeling | Each epoch, predicts on a random slice of unlabeled images and trains on those above 0.95 confidence | same file |
| Cross-dataset eval | Tests the Apple classifier on PlantDoc's overlapping classes; confusion matrices for both | `src/eval/` |
| Class-incremental | Base task (healthy, scab) with full data, then black rot and cedar rust from 20 shots each; naive vs. replay + distillation | `src/incremental/` |
| Grad-CAM boxes | Hand-written Grad-CAM, thresholded into the largest connected region as a pseudo box | `src/interpretability/` |
| Segmentation | U-Net trained with BCE + Dice on masks derived from PlantVillage's background-removed images | `src/training/train_segmentation.py` |
| Synthetic data | Per-class DCGAN at 64×64 for the rarest Apple class | `src/training/train_gan.py` |

### Design decisions

- **Data from the original source, not a Kaggle mirror.** The popular "New Plant Diseases Dataset
  (Augmented)" mirror places augmented copies of the same leaf in both train and validation, which
  inflates validation accuracy. This repo clones the raw images and splits them itself.
- **Report macro-F1, not just accuracy.** PlantVillage classes are imbalanced; macro-F1 weights each disease equally.
- **Pseudo-labels hold file paths, not image tensors,** and train in normal mini-batches, so memory use
  doesn't grow with the unlabeled pool.
- **Class-incremental evaluation masks unseen classes.** After each task the model can only predict
  classes it has been taught, and distillation is applied only over the teacher's known classes.
- **Every training run checkpoints each epoch** and can `--resume`, because Colab sessions disconnect.

### Limitations

- The train/validation split is random per image, not grouped by physical leaf, so some near-duplicate
  photos of one leaf may land on both sides; this is why the PlantDoc result is the more trustworthy number.
- The class-incremental benchmark has only 4 classes and 3 tasks; it shows the method works, not how it scales.
- DCGAN samples are 64×64; a diffusion model would give higher fidelity at much higher compute cost.
- Grad-CAM boxes are not scored against ground-truth boxes (PlantVillage has none).

## Reproduce

### Google Colab (recommended)

Click the **Open in Colab** badge above, select a T4 GPU, and run the cells top to bottom; the notebook
clones this repo itself. See **[docs/COLAB.md](docs/COLAB.md)** for the full walkthrough. No API keys or
accounts are needed beyond Google Drive.

### Local

```bash
git clone https://github.com/tajammulbasheer/plant_leaf_disease_detection.git
cd plant_leaf_disease_detection
pip install -r requirements.txt
python -m src.data.prepare_plantvillage --output-dir data --cleanup-clone

python -m src.training.train_classifier --data-dir data/plantvillage_full \
    --output-dir runs/baseline_full --mode full --epochs 20
# ...other steps follow the notebook; then:
python -m src.utils.generate_report --run-dir runs --output-dir runs/report \
    --assets-dir assets --readme README.md
```

Each script prints its options with `--help`.

## Repository structure

```
src/
  data/
    dataset.py                 class-folder dataset, few-shot sampling, transforms, seeding
    prepare_plantvillage.py    download, split, Apple subset, segmentation masks
  models/
    classifier.py              EfficientNet / ResNet classifier
    unet.py                    U-Net
    gan.py                     DCGAN generator and discriminator
  training/
    train_classifier.py        full / few-shot / semi-supervised, resumable
    train_segmentation.py      U-Net training with IoU / Dice
    train_gan.py               per-class DCGAN
  eval/
    plantdoc.py                PlantDoc loader (class mapping)
    cross_dataset_eval.py      in- vs. out-of-distribution evaluation
  incremental/
    train_incremental.py       class-incremental few-shot learning
    metrics.py                 accuracy matrix, forgetting
  interpretability/
    gradcam.py                 Grad-CAM
    pseudo_boxes.py            heatmap → pseudo bounding box
  utils/
    metrics.py                 accuracy, macro-F1 / precision / recall
    generate_report.py         results tables, figures, README update
notebooks/train_on_colab.ipynb  Colab runner (calls the scripts above)
docs/COLAB.md                   step-by-step Colab guide
assets/                         figures used in this README
```

## Data and credits

- **PlantVillage**: Hughes & Salathé, *An open access repository of images on plant health to enable the
  development of mobile disease diagnostics*, 2015. Images from
  [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset).
- **PlantDoc**: Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020.
  [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset).

Datasets are downloaded at run time and are not redistributed in this repository.

## License

MIT. See [LICENSE](LICENSE).
