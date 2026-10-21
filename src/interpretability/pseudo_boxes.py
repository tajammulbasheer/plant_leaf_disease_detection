"""
Threshold a Grad-CAM heatmap into a pseudo bounding box around the most
activated region, and optionally score it against a small set of
hand-annotated ground-truth boxes (IoU) if you choose to label a handful
of validation images for a sanity-checked localization metric.

This turns a pure classifier into something closer to weakly-supervised
detection: no box labels were used during training, only image-level
disease labels.
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.data.dataset import AppleLeafDataset, APPLE_CLASSES, default_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.classifier import LeafClassifier, load_checkpoint_classes
from src.interpretability.gradcam import GradCAM, get_last_conv_layer


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def cam_to_bbox(cam: torch.Tensor, threshold: float = 0.5):
    """Largest connected component above `threshold` -> (x, y, w, h) box."""
    mask = (cam.numpy() > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:  # only background found
        return None
    # stats[0] is background; pick the largest remaining component by area
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[largest_idx]
    return int(x), int(y), int(w), int(h)


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def visualize(image_denorm, cam, bbox, class_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    img_np = image_denorm.permute(1, 2, 0).numpy()
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(cam.numpy(), cmap="jet", alpha=0.5)
    axes[1].set_title(f"Grad-CAM ({class_name})")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    if bbox is not None:
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, edgecolor="lime", facecolor="none", linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title("Pseudo bounding box")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint_classes(args.checkpoint, APPLE_CLASSES)
    model = LeafClassifier(num_classes=len(APPLE_CLASSES), backbone=args.backbone, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    target_layer = get_last_conv_layer(model)
    cam_extractor = GradCAM(model, target_layer)

    dataset = AppleLeafDataset(
        root=args.data_dir,
        transform=default_transforms(train=False, image_size=args.image_size),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(dataset))
    for i in range(n):
        image, label = dataset[i]
        image_batch = image.unsqueeze(0).to(device)

        cam, pred_class = cam_extractor(image_batch)
        bbox = cam_to_bbox(cam.cpu(), threshold=args.cam_threshold)

        class_name = APPLE_CLASSES[pred_class].replace("Apple___", "")
        visualize(
            denormalize(image.cpu()), cam.cpu(), bbox, class_name,
            out_dir / f"sample_{i}_{class_name}.png",
        )

    print(f"Saved {n} Grad-CAM + pseudo-bbox visualizations to {out_dir}")
    print(
        "Note: no ground-truth boxes were used anywhere in this pipeline -- "
        "these boxes are derived purely from image-level classification labels."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True, help="e.g. .../apple/val")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cam-threshold", type=float, default=0.5)
    p.add_argument("--num-samples", type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
