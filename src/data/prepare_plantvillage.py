"""
Prepare training data by cloning the original PlantVillage-Dataset GitHub
repo (spMohanty/PlantVillage-Dataset) directly, rather than depending on
the Hugging Face mirror -- which has proven unreliable as a dependency for
this pipeline: its config name changed once (from "color" to "default"),
and its current loader schema turns out to build the dataset from three
parallel path-list text files (color/grayscale/segmented merged) rather
than serving actual image+label columns. That's a moving target that isn't
worth chasing.

The raw GitHub repo, by contrast, is a plain folder-per-class structure
that hasn't changed in years:

    PlantVillage-Dataset/raw/color/<class_name>/*.JPG

There's no pre-baked train/val split in the raw repo, so this script
performs its own stratified 80/20 split (seeded, reproducible). This is
still markedly lower-risk than the Kaggle "New Plant Diseases Dataset
(Augmented)" mirror's built-in split, since these are single, non-augmented
photos -- there's no near-duplicate-from-augmentation leakage here, which
was the actual mechanism behind the ~100% validation accuracy you saw
earlier. It is a random split rather than a leaf-grouping-aware one, and
that distinction is noted in the printed output and should be noted in
your README's methodology section.

It also derives leaf/background masks for the U-Net segmentation step from
the repo's `raw/segmented/` folder (the same photos with the background
blacked out), pairing each with its colour original:

    <output-dir>/seg/images/*.jpg   colour photos (Apple classes)
    <output-dir>/seg/masks/*.png    binary masks, 255 = leaf, 0 = background

Example (Colab):
    !python -m src.data.prepare_plantvillage --output-dir /content/data --cleanup-clone
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
APPLE_CLASS_KEYWORDS = ("apple",)


def clone_repo(dest: Path):
    if (dest / "raw" / "color").exists():
        print(f"{dest} already exists, skipping clone.")
        return
    if dest.exists():
        print(f"{dest} looks incomplete (earlier clone interrupted?) -- removing and re-cloning.")
        shutil.rmtree(dest)
    print(f"Cloning {REPO_URL} (shallow clone, ~1-2GB -- give it a few minutes)...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)


def split_and_copy(class_name: str, files, full_root: Path, apple_root: Path, is_apple: bool):
    if len(files) < 2:
        return
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        dst_dir = full_root / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy(f, dst_dir / f.name)

        if is_apple:
            apple_dst = apple_root / split_name / class_name
            apple_dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, apple_dst / f.name)


def _norm_stem(name: str) -> str:
    """Match a segmented file to its colour original: the segmented copies
    carry a suffix such as '_final_masked'; strip it and compare case-insensitively."""
    stem = Path(name).stem.lower()
    for suffix in ("_final_masked", "_masked", "_segmented"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def mask_from_segmented(seg_path: Path, threshold: int = 20):
    """Binary leaf mask from a background-blacked-out image."""
    img = cv2.imread(str(seg_path))
    if img is None:
        return None
    mask = (img.max(axis=2) > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Keep the largest connected component (the leaf), drop speckle.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask * 255


def build_seg_masks(repo_dir: Path, out_root: Path, class_filter=APPLE_CLASS_KEYWORDS):
    color_dir = repo_dir / "raw" / "color"
    seg_dir = repo_dir / "raw" / "segmented"
    if not seg_dir.exists():
        print(f"WARNING: {seg_dir} not found -- skipping segmentation masks.")
        return 0
    img_out, mask_out = out_root / "images", out_root / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for class_dir in sorted(d for d in seg_dir.iterdir() if d.is_dir()):
        if not class_dir.name.lower().startswith(class_filter):
            continue
        color_lookup = {
            _norm_stem(f.name): f for f in (color_dir / class_dir.name).iterdir()
        } if (color_dir / class_dir.name).exists() else {}
        for seg_file in tqdm(sorted(class_dir.iterdir()), desc=f"Masks {class_dir.name}"):
            color_file = color_lookup.get(_norm_stem(seg_file.name))
            if color_file is None:
                continue
            mask = mask_from_segmented(seg_file)
            if mask is None or mask.mean() < 1:  # empty mask -> skip
                continue
            out_stem = f"{class_dir.name}__{written:05d}"
            shutil.copy(color_file, img_out / f"{out_stem}.jpg")
            cv2.imwrite(str(mask_out / f"{out_stem}.png"), mask)
            written += 1
    if written == 0:
        print("WARNING: no segmented/colour pairs matched; check file naming in raw/segmented.")
    else:
        print(f"Wrote {written} image/mask pairs to {out_root}")
    return written


def run(args):
    repo_dir = Path(args.workdir) / "PlantVillage-Dataset"
    clone_repo(repo_dir)

    color_dir = repo_dir / "raw" / "color"
    if not color_dir.exists():
        raise RuntimeError(
            f"Expected {color_dir} to exist after cloning; the repo layout may "
            f"have changed. Check {REPO_URL} manually."
        )

    class_names = sorted([d.name for d in color_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class subfolders found under {color_dir}.")
    print(f"Found {len(class_names)} classes.")

    full_root = Path(args.output_dir) / "plantvillage_full"
    apple_root = Path(args.output_dir) / "apple"

    for class_name in tqdm(class_names, desc="Splitting + copying classes"):
        class_dir = color_dir / class_name
        files = sorted(
            f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        is_apple = class_name.lower().startswith(APPLE_CLASS_KEYWORDS)
        split_and_copy(class_name, files, full_root, apple_root, is_apple)

    print(f"\nFull dataset ({len(class_names)} classes) written to: {full_root}")
    print(f"Apple-only subset written to: {apple_root}")
    print(
        "Split performed locally (stratified 80/20 by class, seed=42) on "
        "non-augmented source images -- no augmentation-duplicate leakage."
    )

    if not args.no_seg_masks:
        build_seg_masks(repo_dir, Path(args.output_dir) / "seg")

    if args.cleanup_clone:
        print(f"Removing cloned repo at {repo_dir} to free disk space...")
        shutil.rmtree(repo_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, help="e.g. /content/data")
    p.add_argument("--workdir", default="/content", help="Where to clone the repo (temporary, separate from output-dir)")
    p.add_argument("--cleanup-clone", action="store_true", help="Delete the cloned repo after copying files out of it")
    p.add_argument("--no-seg-masks", action="store_true", help="Skip building U-Net masks from raw/segmented")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
