"""
Grad-CAM for the LeafClassifier, used to localize disease lesions without
any bounding box supervision -- the model only ever sees image-level labels,
but Grad-CAM shows *where* in the image it's basing its decision on, which we
then threshold into a pseudo bounding box (see pseudo_boxes.py).

Implemented by hand (hooks into the last conv layer) rather than pulling in
a Grad-CAM library, so it stays dependency-light and works with the
EfficientNet/ResNet backbones already in this repo without modification.
"""

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image: torch.Tensor, class_idx: int = None):
        """
        Args:
            image: (1, 3, H, W) single normalized image tensor, requires no grad set beforehand.
            class_idx: which class to explain. Defaults to the model's top prediction.

        Returns:
            cam: (H, W) numpy-free torch tensor, values in [0, 1], resized to input resolution.
            class_idx: the class index that was explained.
        """
        self.model.zero_grad()
        image = image.clone().requires_grad_(True)
        logits = self.model(image)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach(), class_idx


def get_last_conv_layer(model):
    """
    Returns the last convolutional layer of the classifier's backbone --
    the standard Grad-CAM target. Works for both EfficientNet and ResNet
    backbones as wrapped by LeafClassifier.
    """
    backbone = model.backbone
    if hasattr(backbone, "features"):  # EfficientNet
        return backbone.features[-1]
    if hasattr(backbone, "layer4"):  # ResNet
        return backbone.layer4[-1]
    raise ValueError("Unrecognized backbone architecture; add a case in get_last_conv_layer().")
