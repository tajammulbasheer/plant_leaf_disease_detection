"""
Train a per-class DCGAN to generate synthetic images for class balancing.

Example (Colab):
    !python -m src.training.train_gan \
        --data-dir /content/data/apple/train/Apple___Cedar_apple_rust \
        --output-dir /content/drive/MyDrive/apple_leaf_runs/gan_cedar_rust \
        --epochs 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from src.data.dataset import PathListDataset, set_seed
import torchvision.transforms as T
from tqdm import tqdm

from src.models.gan import Generator, Discriminator, weights_init


def build_loader(data_dir, image_size, batch_size):
    """Loader over the images of ONE class folder.

    (An earlier version pointed ImageFolder at the parent directory, which
    silently trained the GAN on every class in it, not just this one.)
    """
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    files = sorted(
        p for p in Path(data_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(files) < batch_size:
        raise RuntimeError(f"Need at least {batch_size} images in {data_dir}, found {len(files)}.")
    print(f"Training GAN on {len(files)} images from {data_dir}")
    dataset = PathListDataset([(str(f), 0) for f in files], transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(args.data_dir, args.image_size, args.batch_size)

    netG = Generator(latent_dim=args.latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, args.latent_dim, 1, 1, device=device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for real, _ in tqdm(loader, desc=f"GAN epoch {epoch+1}/{args.epochs}"):
            real = real.to(device)
            b_size = real.size(0)
            real_labels = torch.full((b_size,), 0.9, device=device)  # label smoothing
            fake_labels = torch.zeros(b_size, device=device)

            # --- Discriminator step ---
            optD.zero_grad()
            out_real = netD(real)
            loss_d_real = criterion(out_real, real_labels)

            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            out_fake = netD(fake.detach())
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # --- Generator step ---
            optG.zero_grad()
            out_fake_for_g = netD(fake)
            loss_g = criterion(out_fake_for_g, torch.full((b_size,), 1.0, device=device))
            loss_g.backward()
            optG.step()

        print(f"Epoch {epoch+1}: loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f}")

        if (epoch + 1) % args.sample_every == 0:
            with torch.no_grad():
                samples = netG(fixed_noise).cpu()
            vutils.save_image(samples, out_dir / f"samples_epoch{epoch+1}.png", normalize=True, nrow=4)

    torch.save(netG.state_dict(), out_dir / "generator.pt")

    # Side-by-side real vs generated grid for the README.
    real_batch, _ = next(iter(loader))
    with torch.no_grad():
        fake_batch = netG(torch.randn(8, args.latent_dim, 1, 1, device=device)).cpu()
    grid = torch.cat([real_batch[:8], fake_batch], dim=0)
    vutils.save_image(grid, out_dir / "real_vs_generated.png", normalize=True, nrow=8)
    print(f"Saved real (top row) vs generated (bottom row) grid to {out_dir / 'real_vs_generated.png'}")
    print(f"Saved generator to {out_dir / 'generator.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Path to the single class folder to model")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
