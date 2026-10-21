"""
Loader for the PlantDoc dataset, used purely as an out-of-distribution
(OOD) test set to measure how well a PlantVillage-trained model generalizes
to real-world field photos (messy backgrounds, variable lighting, occlusion)
vs. the lab-condition photos PlantVillage is made of.

PlantDoc source: https://github.com/pratikkayal/PlantDoc-Dataset
(Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", 2020)

PlantDoc's apple-relevant classes don't line up 1:1 with PlantVillage's
naming, and it has no direct equivalent for "Black_rot". We evaluate only
on the classes that DO overlap, and report that limitation explicitly
rather than silently dropping it.
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from PIL import Image

# Maps PlantDoc's folder names -> PlantVillage's Apple___* label indices.
# "Black_rot" is intentionally absent: PlantDoc has no matching class.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",
}


class PlantDocAppleDataset(Dataset):
    """
    Expects PlantDoc's raw folder structure:
        root/<class_name>/*.jpg

    Only folders present in PLANTDOC_TO_PLANTVILLAGE are used; everything
    else (grape, tomato, etc.) is ignored.
    """

    def __init__(self, root: str, class_to_idx: dict, transform: Optional[object] = None):
        """
        Args:
            root: path to PlantDoc's test (or train) directory.
            class_to_idx: mapping from PlantVillage class name -> label index,
                e.g. {name: i for i, name in enumerate(APPLE_CLASSES)}. Passed
                in explicitly so label indices always match the trained model.
            transform: torchvision transform pipeline (use the classifier's
                eval-time `default_transforms(train=False)`).
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.skipped_classes = []

        for pd_name, pv_name in PLANTDOC_TO_PLANTVILLAGE.items():
            if pv_name not in class_to_idx:
                continue
            class_dir = self.root / pd_name
            if not class_dir.exists():
                self.skipped_classes.append(pd_name)
                continue
            label = class_to_idx[pv_name]
            for f in class_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(f), label))

        if not self.samples:
            raise RuntimeError(
                f"No matching PlantDoc classes found under {self.root}. "
                f"Expected one of {list(PLANTDOC_TO_PLANTVILLAGE.keys())}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def missing_classes_note(self) -> str:
        note = "Note: 'Black_rot' has no PlantDoc equivalent and is excluded from this OOD eval."
        if self.skipped_classes:
            note += f" Also missing on disk: {self.skipped_classes}."
        return note
