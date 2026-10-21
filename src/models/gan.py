"""
A lightweight DCGAN used to generate synthetic leaf images for
under-represented disease classes (class balancing), as an alternative to
naive oversampling.

Train one generator/discriminator pair per minority class on a 64x64
crop resolution (fast on a single Colab T4/A100). Generated images are
upsampled back to the classifier's input resolution and mixed into the
training set at a configurable ratio -- see
`src/training/train_classifier.py --synthetic-ratio`.

Note: for higher-fidelity augmentation, a small diffusion model
(e.g. a distilled DDPM) can be swapped in -- this DCGAN is intentionally
lightweight so the whole augmentation step fits in a single Colab session.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_channels: int = 3, feat_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feat_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 8, feat_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 4, feat_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps * 2, feat_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, feat_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_maps, feat_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_maps * 2, feat_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_maps * 4, feat_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
