"""
Disease classifier: a transfer-learning backbone (default: EfficientNet-B0)
fine-tuned on the segmented leaf crops.

Kept intentionally simple/swappable -- pass a different `backbone` name to
try ResNet18/34 or other torchvision models with a matching feature dim.
"""

import torch.nn as nn
import torchvision.models as tvm


BACKBONES = {
    "efficientnet_b0": (tvm.efficientnet_b0, 1280),
    "resnet18": (tvm.resnet18, 512),
    "resnet50": (tvm.resnet50, 2048),
}


class LeafClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONES)}")

        ctor, feat_dim = BACKBONES[backbone]
        weights = "DEFAULT" if pretrained else None
        base = ctor(weights=weights)

        if backbone.startswith("efficientnet"):
            base.classifier = nn.Identity()
        else:  # resnet family
            base.fc = nn.Identity()

        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def load_checkpoint_classes(checkpoint_path, expected_classes=None):
    """Read the classes.json saved next to a checkpoint by train_classifier.

    If `expected_classes` is given and doesn't match, raise a clear error
    instead of PyTorch's opaque size-mismatch message (the usual cause:
    passing the 38-class PlantVillage checkpoint to an Apple-only script).
    """
    import json
    from pathlib import Path

    classes_path = Path(checkpoint_path).parent / "classes.json"
    if not classes_path.exists():
        return expected_classes
    classes = json.loads(classes_path.read_text())
    if expected_classes is not None and list(classes) != list(expected_classes):
        raise ValueError(
            f"{checkpoint_path} was trained on {len(classes)} classes, but this script "
            f"needs the {len(expected_classes)} Apple classes {list(expected_classes)}.\n"
            f"Train one with:  python -m src.training.train_classifier "
            f"--data-dir <data>/apple --output-dir <runs>/baseline_apple_only --mode full\n"
            f"and pass <runs>/baseline_apple_only/best_model.pt instead."
        )
    return classes
