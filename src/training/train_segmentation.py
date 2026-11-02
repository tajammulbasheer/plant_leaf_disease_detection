"""
Train the U-Net leaf/background segmentation model.

Expects a directory of (image, mask) pairs:
    images/*.jpg
    masks/*.png   (binary: 255 = leaf, 0 = background)

Masks are produced automatically by src/data/prepare_plantvillage.py from
the PlantVillage repo's raw/segmented/ images (written to <data>/seg/).

Outputs: unet_best.pt, seg_results.json (best val IoU / Dice),
seg_history.json, and seg_samples.png (input | ground truth | prediction).

Example (Colab):
    !python -m src.training.train_segmentation \
        --images-dir /content/data/seg/images \
        --masks-dir /content/data/seg/masks \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/unet \
        --epochs 30
"""

import argparse
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from src.models.unet import UNet
from src.data.dataset import set_seed


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=256):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        masks = {p.stem: p for p in self.masks_dir.iterdir()}
        self.pairs = [
            (p, masks[p.stem]) for p in sorted(self.images_dir.iterdir()) if p.stem in masks
        ]
        if not self.pairs:
            raise RuntimeError(f"No matching image/mask pairs in {images_dir} and {masks_dir}.")
        self.image_size = image_size
        self.img_transform = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor(),
             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.mask_transform = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor()]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def iou_and_dice(pred_logits, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return iou.sum().item(), dice.sum().item()


def save_samples(model, dataset, device, path, n=4):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    model.eval()
    with torch.no_grad():
        for i in range(n):
            image, mask = dataset[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0, 0].cpu() > 0.5
            img = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            for ax, data, title in zip(axes[i], [img, mask[0].numpy(), pred.numpy()],
                                       ["Input", "Ground-truth mask", "Predicted mask"]):
                ax.imshow(data, cmap=None if data.ndim == 3 else "gray")
                ax.set_title(title)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(args.images_dir, args.masks_dir, args.image_size)
    val_len = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    best_val_loss = float("inf")
    best = {}
    history = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            if scaler is None:
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                loss.backward()
                optimizer.step()
            else:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images)
                # Loss in fp32: Dice sums over every pixel and can overflow fp16.
                logits = logits.float()
                loss = bce(logits, masks) + dice_loss(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss, iou_sum, dice_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * images.size(0)
                i_sum, d_sum = iou_and_dice(logits, masks)
                iou_sum += i_sum
                dice_sum += d_sum

        train_loss = running_loss / len(train_ds)
        val_loss = val_loss / len(val_ds)
        val_iou, val_dice = iou_sum / len(val_ds), dice_sum / len(val_ds)
        print(f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_IoU={val_iou:.4f} val_Dice={val_dice:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "val_iou": val_iou, "val_dice": val_dice})
        (out_dir / "seg_history.json").write_text(json.dumps(history, indent=2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = {"epoch": epoch + 1, "val_iou": val_iou, "val_dice": val_dice,
                    "val_loss": val_loss, "n_train": len(train_ds), "n_val": len(val_ds),
                    "amp": args.amp}
            torch.save(model.state_dict(), out_dir / "unet_best.pt")
            (out_dir / "seg_results.json").write_text(json.dumps(best, indent=2))

    model.load_state_dict(torch.load(out_dir / "unet_best.pt", map_location=device))
    save_samples(model, val_ds, device, out_dir / "seg_samples.png")
    print(f"Best epoch {best.get('epoch')}: IoU={best.get('val_iou', 0):.4f} "
          f"Dice={best.get('val_dice', 0):.4f}. Outputs in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision (fp16 autocast + GradScaler) training steps; validation stays fp32")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
