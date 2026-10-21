"""
Dataset utilities for leaf disease classification.

Works with any class-subfoldered image directory:

    root/
        <class_a>/*.jpg
        <class_b>/*.jpg
        ...

Used both for the full 38-class PlantVillage dataset and for the Apple-only
4-class subset used by the segmentation/GAN/incremental/interpretability
extensions -- the loader doesn't care which, it discovers classes from the
directory structure unless an explicit class list is passed in.

Data is expected to come from `src/data/prepare_plantvillage.py`, which
pulls the leaf-grouping-safe official split (no train/val leakage) rather
than the commonly-used Kaggle "augmented" mirror, which has known
duplicate-leaf leakage between its train/valid folders.
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Kept for backward compatibility with the Apple-only extension scripts
# (segmentation, GAN, incremental learning, Grad-CAM, PlantDoc cross-eval),
# which intentionally stay scoped to these 4 classes.
APPLE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def discover_classes(root: str) -> List[str]:
    """Sorted list of immediate subdirectory names -- i.e. the class list."""
    root = Path(root)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def default_transforms(image_size: int = 224, train: bool = True):
    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LeafDataset(Dataset):
    """
    Expects a directory structure:

        root/<class_name>/*.jpg

    Args:
        root: path to the class-subfoldered image directory.
        transform: torchvision transform pipeline. Defaults to
            `default_transforms(train=True)`.
        k_shot: if set, randomly sample only `k_shot` images per class
            (few-shot setting). Sampling is seeded for reproducibility.
        seed: random seed used for few-shot sampling.
        classes: restrict to this subset of class names (used by the
            class-incremental experiments, and by anything working with
            the Apple-only subset). Defaults to every subfolder found
            under `root` -- i.e. all 38 classes for the full dataset.
        class_to_idx: explicit label-index mapping to use, so label
            indices stay consistent across train/val calls (and across
            incremental tasks with a fixed-size head). Defaults to
            enumerate(classes).
    """

    def __init__(
        self,
        root: str,
        transform: Optional[T.Compose] = None,
        k_shot: Optional[int] = None,
        seed: int = 42,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform or default_transforms(train=True)
        self.samples = []

        self.classes = classes if classes is not None else discover_classes(self.root)
        self.class_to_idx = class_to_idx or {name: i for i, name in enumerate(self.classes)}

        rng = random.Random(seed)
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            files = sorted(
                [
                    p
                    for p in class_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
            )
            if k_shot is not None and len(files) > k_shot:
                files = rng.sample(files, k_shot)
            self.samples.extend([(str(f), class_idx) for f in files])

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. Check the dataset path "
                f"and that class subfolders match {self.classes}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

    def class_counts(self) -> Dict[str, int]:
        idx_to_name = {i: name for name, i in self.class_to_idx.items()}
        counts = {name: 0 for name in self.classes}
        for _, label in self.samples:
            counts[idx_to_name[label]] += 1
        return counts


# Backward-compatible alias: earlier versions of this repo only supported
# the Apple subset and named the class accordingly.
AppleLeafDataset = LeafDataset


class PathListDataset(Dataset):
    """Dataset over an explicit list of (image_path, label) pairs.

    Used for pseudo-labeled samples (semi-supervised mode) and for the
    exemplar replay buffer (class-incremental mode). Storing paths rather
    than decoded image tensors keeps host memory flat no matter how many
    samples are selected.
    """

    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class IndexedDataset(Dataset):
    """Wraps a dataset so each item also returns its index (to map
    predictions back to file paths without holding images in memory)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        return image, idx


def set_seed(seed: int = 42):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

