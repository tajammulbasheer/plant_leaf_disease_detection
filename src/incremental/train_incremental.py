"""
Class-incremental few-shot learning on the apple leaf classes.

Simulates diseases being "discovered" over time: the model first learns a
base set of classes with full data, then incrementally learns new classes
from only `k` labeled examples each, one task at a time -- without access
to full old-task data going forward (only a small exemplar buffer, if the
'replay_distill' strategy is used).

Two strategies, run back-to-back so their forgetting can be compared directly:

  naive          -- fine-tune on each new task's data only. Expected to show
                    significant catastrophic forgetting on earlier classes.
  replay_distill -- keep a small exemplar buffer (k images/class) from every
                    previously learned class, replay them each batch, and add
                    a knowledge-distillation loss (soft targets from the
                    pre-task model) on those exemplars to preserve old
                    decision boundaries.

Default task split (change via --task-order):
  Task 0 (base, full data): healthy, Apple_scab
  Task 1 (k-shot):          Black_rot
  Task 2 (k-shot):          Cedar_apple_rust

Example (Colab):
    !python -m src.incremental.train_incremental \
        --data-dir /content/data/apple \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/incremental \
        --k-shot 20 --epochs-per-task 15
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataset import AppleLeafDataset, APPLE_CLASSES, PathListDataset, default_transforms, set_seed
from src.models.classifier import LeafClassifier
from src.incremental.metrics import compute_forgetting, average_incremental_accuracy

DEFAULT_TASK_ORDER = [
    ["Apple___healthy", "Apple___Apple_scab"],  # task 0: base classes, full data
    ["Apple___Black_rot"],                       # task 1: new class, k-shot
    ["Apple___Cedar_apple_rust"],                 # task 2: new class, k-shot
]

CLASS_TO_IDX = {name: i for i, name in enumerate(APPLE_CLASSES)}


def build_task_loader(data_dir, task_classes, image_size, batch_size, k_shot=None, train=True):
    ds = AppleLeafDataset(
        root=f"{data_dir}/{'train' if train else 'val'}",
        transform=default_transforms(train=train, image_size=image_size),
        classes=task_classes,
        class_to_idx=CLASS_TO_IDX,
        k_shot=k_shot,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2), ds


def sample_exemplars(dataset, n_per_class, seed=42):
    """Randomly pick `n_per_class` samples per class for the replay buffer."""
    import random as _random
    rng = _random.Random(seed)
    by_class = {}
    for path, label in dataset.samples:
        by_class.setdefault(label, []).append(path)
    exemplars = []
    for label, paths in sorted(by_class.items()):
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        exemplars.extend([(p, label) for p in chosen])
    return exemplars


def evaluate_task(model, data_dir, task_classes, seen_classes, image_size, batch_size, device):
    """Accuracy on one task's val images, predicting only among the classes
    learned so far (standard class-incremental evaluation: the head has
    slots for future classes, but they must not be predictable yet)."""
    loader, _ = build_task_loader(data_dir, task_classes, image_size, batch_size, train=False)
    seen_idx = torch.tensor([CLASS_TO_IDX[c] for c in seen_classes], device=device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            masked = torch.full_like(logits, float("-inf"))
            masked[:, seen_idx] = logits[:, seen_idx]
            preds = masked.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else float("nan")


def run_strategy(args, strategy: str):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeafClassifier(num_classes=len(APPLE_CLASSES), backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    task_order = DEFAULT_TASK_ORDER
    num_tasks = len(task_order)
    accuracy_matrix = np.full((num_tasks, num_tasks), np.nan)

    exemplar_buffer = []  # list of (path, label), grows after each task
    teacher_model = None

    for task_idx, task_classes in enumerate(task_order):
        is_base_task = task_idx == 0
        k_shot = None if is_base_task else args.k_shot

        train_loader, train_ds = build_task_loader(
            args.data_dir, task_classes, args.image_size, args.batch_size, k_shot=k_shot, train=True
        )

        exemplar_loader = None
        if strategy == "replay_distill" and exemplar_buffer:
            exemplar_ds = PathListDataset(
                exemplar_buffer, default_transforms(train=True, image_size=args.image_size)
            )
            exemplar_loader = DataLoader(exemplar_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        # Fresh iterator for every task, so this task's replay draws from the
        # buffer as it now stands (including exemplars added after the
        # previous task). Previously the old iterator leaked across tasks.
        exemplar_iter = iter(exemplar_loader) if exemplar_loader is not None else None
        old_idx = torch.tensor(
            [CLASS_TO_IDX[c] for t in task_order[:task_idx] for c in t], dtype=torch.long
        )

        print(f"\n=== Strategy '{strategy}' — Task {task_idx}: {task_classes} "
              f"({'full data' if is_base_task else f'{args.k_shot}-shot'}) ===")

        for epoch in range(args.epochs_per_task):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)

                # Exemplar replay + distillation: also train on a batch of old
                # exemplars, matching the teacher's soft predictions on them.
                if exemplar_loader is not None:
                    try:
                        ex_images, ex_labels = next(exemplar_iter)
                    except StopIteration:
                        exemplar_iter = iter(exemplar_loader)
                        ex_images, ex_labels = next(exemplar_iter)
                    ex_images, ex_labels = ex_images.to(device), ex_labels.to(device)

                    ex_logits = model(ex_images)
                    replay_ce = criterion(ex_logits, ex_labels)

                    with torch.no_grad():
                        teacher_logits = teacher_model(ex_images)
                    # Distill only over the classes the teacher actually knew.
                    oi = old_idx.to(device)
                    distill_loss = F.kl_div(
                        F.log_softmax(ex_logits[:, oi] / args.temperature, dim=1),
                        F.softmax(teacher_logits[:, oi] / args.temperature, dim=1),
                        reduction="batchmean",
                    ) * (args.temperature ** 2)

                    loss = loss + args.replay_weight * replay_ce + args.distill_weight * distill_loss

                loss.backward()
                optimizer.step()

        # --- Evaluate on every task seen so far ---
        cumulative_classes = [c for t in task_order[: task_idx + 1] for c in t]
        for eval_idx in range(task_idx + 1):
            acc = evaluate_task(
                model, args.data_dir, task_order[eval_idx], cumulative_classes,
                args.image_size, args.batch_size, device,
            )
            accuracy_matrix[task_idx, eval_idx] = acc
        print(f"  Row {task_idx} accuracies: {accuracy_matrix[task_idx, :task_idx+1]}")

        # --- Update exemplar buffer and teacher snapshot for next task ---
        if strategy == "replay_distill":
            exemplar_buffer.extend(sample_exemplars(train_ds, args.exemplars_per_class, seed=args.seed))
            teacher_model = copy.deepcopy(model).eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

    forgetting = compute_forgetting(accuracy_matrix)
    avg_incremental_acc = average_incremental_accuracy(accuracy_matrix)
    return {
        "strategy": strategy,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "average_forgetting": forgetting,
        "final_average_accuracy": avg_incremental_acc,
    }


def run(args):
    results = {}
    for strategy in ["naive", "replay_distill"]:
        results[strategy] = run_strategy(args, strategy)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "incremental_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for strategy, r in results.items():
        print(
            f"{strategy}: final_avg_acc={r['final_average_accuracy']:.4f} "
            f"avg_forgetting={r['average_forgetting']:.4f}"
        )
    print(f"\nSaved full results to {out_dir / 'incremental_results.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs-per-task", type=int, default=15)
    p.add_argument("--k-shot", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--exemplars-per-class", type=int, default=15)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
