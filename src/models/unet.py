"""
A compact U-Net used to segment the leaf region from background before
classification. Segmenting out the background reduces the classifier's
sensitivity to lighting/soil/background clutter in field photos.

This is trained separately (see src/training/train_segmentation.py) on a
small hand-labeled or weakly-supervised (e.g. GrabCut-bootstrapped) mask set,
then frozen and used as a preprocessing step ahead of the classifier.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Standard 4-level U-Net for binary (leaf vs. background) segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def apply_mask(image: torch.Tensor, mask_logits: torch.Tensor, threshold: float = 0.5):
    """Zero out background pixels using a predicted segmentation mask.

    Args:
        image: (B, 3, H, W) input image batch, already normalized.
        mask_logits: (B, 1, H, W) raw U-Net output (pre-sigmoid).
        threshold: probability threshold for foreground.
    """
    mask = torch.sigmoid(mask_logits) > threshold
    return image * mask.float()
