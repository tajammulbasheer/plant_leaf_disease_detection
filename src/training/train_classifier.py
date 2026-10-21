"""
Train the leaf disease classifier.

Modes:
  --mode full             standard supervised training on the full labeled set
  --mode few_shot          train with only `--k-shot` labeled images per class
  --mode semi_supervised   `--k-shot` labeled images per class, plus
                           confidence-thresholded pseudo-labels drawn each
                           epoch from a random slice of the remaining
                           (unlabeled) training pool

Outputs written to --output-dir (updated after EVERY epoch, so an
interrupted run keeps its progress; re-run with --resume to continue):
  history.json      per-epoch train loss, val accuracy / macro-F1 / P / R
  best_model.pt     weights of the best epoch by val macro-F1
  best_epoch.json   which epoch that was
  classes.json      class names, in label-index order
  config.json       the exact arguments used for the run

Example (Colab):
    !python -m src.training.train_classifier \
        --data-dir /content/data/plantvillage_full \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/semi_supervised_k20 \
        --mode semi_supervised --k-shot 20 --epochs 25 --resume
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import (
    IndexedDataset,
    LeafDataset,
    PathListDataset,
    default_transforms,
    discover_classes,
    set_seed,
)
from src.models.classifier import LeafClassifier
from src.utils.metrics import evaluate_classifier


def build_dataloaders(args):
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    classes = discover_classes(train_dir)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    use_k_shot = args.mode in ("few_shot", "semi_supervised")

    train_ds = LeafDataset(
        root=train_dir,
        transform=default_transforms(train=True, image_size=args.image_size),
        k_shot=args.k_shot if use_k_shot else None,
        classes=classes,
        class_to_idx=class_to_idx,
        seed=args.seed,
    )
    val_ds = LeafDataset(
        root=val_dir,
        transform=default_transforms(train=False, image_size=args.image_size),
        classes=classes,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    unlabeled_ds = None
    if args.mode == "semi_supervised":
        # Unlabeled pool = every training image NOT in the labeled k-shot
        # subset. Its ground-truth labels are never used for training.
        full_pool = LeafDataset(
            root=train_dir,
            transform=default_transforms(train=False, image_size=args.image_size),
            classes=classes,
            class_to_idx=class_to_idx,
        )
        labeled_paths = {p for p, _ in train_ds.samples}
        keep = [i for i, (p, _) in enumerate(full_pool.samples) if p not in labeled_paths]
        unlabeled_ds = Subset(full_pool, keep)

    return train_ds, train_loader, val_loader, unlabeled_ds, classes


def select_pseudo_labels(model, unlabeled_ds, device, args):
    """Predict on a random slice of the unlabeled pool and return the
    confident predictions as (path, pseudo_label) pairs.

    Only file paths are kept -- never decoded image tensors -- so host and
    GPU memory stay flat regardless of the pool size. (An earlier version
    held every confident image as a tensor and then ran them through the
    network in a single batch, which exhausts Colab RAM / GPU memory at
    full-dataset scale and gets the process killed.)
    """
    n = len(unlabeled_ds)
    limit = n if args.max_unlabeled_per_epoch < 0 else min(args.max_unlabeled_per_epoch, n)
    chosen = torch.randperm(n)[:limit].tolist()

    loader = DataLoader(
        IndexedDataset(Subset(unlabeled_ds, chosen)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # Subset(Subset(full_pool, keep), chosen): map back to file paths.
    base = unlabeled_ds.dataset
    base_indices = [unlabeled_ds.indices[i] for i in chosen]

    model.eval()
    selected = []
    with torch.no_grad():
        for images, local_idx in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            conf, preds = probs.max(dim=1)
            for li, c, p in zip(local_idx.tolist(), conf.tolist(), preds.tolist()):
                if c >= args.confidence_threshold:
                    path, _ = base.samples[base_indices[li]]
                    selected.append((path, p))
    model.train()
    return selected, limit


def run_epoch(model, loader, criterion, optimizer, device, desc):
    model.train()
    total_loss, total = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, train_loader, val_loader, unlabeled_ds, classes = build_dataloaders(args)
    print(f"Classes: {len(classes)} | labeled training images: {len(train_ds)}")
    if unlabeled_ds is not None:
        cap = "all" if args.max_unlabeled_per_epoch < 0 else args.max_unlabeled_per_epoch
        print(f"Unlabeled pool: {len(unlabeled_ds)} images; pseudo-labeling {cap} per epoch.")

    model = LeafClassifier(num_classes=len(classes), backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    history, best_f1, start_epoch = [], 0.0, 0
    history_path, ckpt_path = out_dir / "history.json", out_dir / "best_model.pt"

    if args.resume and history_path.exists() and ckpt_path.exists():
        history = json.loads(history_path.read_text())
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        best_f1 = max((h["macro_f1"] for h in history), default=0.0)
        start_epoch = len(history)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming: {start_epoch} epoch(s) done, best macro-F1 so far {best_f1:.4f}.")
        if start_epoch >= args.epochs:
            print("Run already complete -- nothing to do.")
            return
    elif args.resume:
        print("--resume set, but no previous run found here -- starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        tag = f"Epoch {epoch + 1}/{args.epochs}"
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, tag)
        n_pseudo = 0

        if args.mode == "semi_supervised" and epoch >= args.warmup_epochs:
            selected, scanned = select_pseudo_labels(model, unlabeled_ds, device, args)
            n_pseudo = len(selected)
            if selected:
                pseudo_loader = DataLoader(
                    PathListDataset(selected, default_transforms(train=True, image_size=args.image_size)),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                )
                run_epoch(model, pseudo_loader, criterion, optimizer, device, f"{tag} [pseudo]")
            print(f"  pseudo-labels: {n_pseudo}/{scanned} above confidence {args.confidence_threshold}")

        scheduler.step()
        metrics = evaluate_classifier(model, val_loader, device, classes)
        metrics.update({"epoch": epoch + 1, "train_loss": train_loss, "n_pseudo_labels": n_pseudo})
        history.append(metrics)
        print(f"{tag}: loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f} "
              f"val_f1={metrics['macro_f1']:.4f}")

        history_path.write_text(json.dumps(history, indent=2))
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)
            (out_dir / "best_epoch.json").write_text(
                json.dumps({"epoch": epoch + 1, "macro_f1": best_f1}, indent=2)
            )

    print(f"Done. Best val macro-F1: {best_f1:.4f}. Outputs in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Folder containing train/ and val/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["full", "few_shot", "semi_supervised"], default="full")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--k-shot", type=int, default=20)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Supervised-only epochs before pseudo-labeling starts")
    p.add_argument("--confidence-threshold", type=float, default=0.95)
    p.add_argument("--max-unlabeled-per-epoch", type=int, default=2000,
                   help="Unlabeled images scanned per epoch; -1 for the whole pool")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="Continue from history.json/best_model.pt in --output-dir if present")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
