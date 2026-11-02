"""
Measure training throughput (images/sec) and peak GPU memory with and without --amp, on real
prepared data, using the same training-step code as the real scripts. Nothing is saved; no weights
are trained to completion. The numbers only describe speed on this machine, not model quality.

    .venv/bin/python scripts/benchmark_amp.py --data-dir ~/data/plantvillage
    .venv/bin/python scripts/benchmark_amp.py --data-dir ~/data/plantvillage --mode loader

--mode gpu    (default) one real batch is decoded once and reused, so the number is the GPU step
              cost alone, unaffected by data loading or other CPU load
--mode loader batches come from the real DataLoader (end-to-end, includes JPEG decoding)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import LeafDataset, default_transforms, discover_classes, set_seed  # noqa: E402
from src.models.classifier import LeafClassifier  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.training.train_classifier import run_epoch  # noqa: E402
from src.training.train_segmentation import SegmentationDataset, dice_loss  # noqa: E402


def batches(loader, n, mode):
    """n batches: fresh from the loader, or one fixed batch repeated (mode='gpu')."""
    if mode == "gpu":
        batch = next(iter(loader))
        return itertools.repeat(batch, n)
    return itertools.islice(iter(loader), n)


def timed(fn, device, n_images):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return n_images / dt, torch.cuda.max_memory_allocated(device) / 2**20


def bench_classifier(args, amp, device):
    set_seed(42)
    train_dir = Path(args.data_dir) / "plantvillage_full" / "train"
    classes = discover_classes(train_dir)
    ds = LeafDataset(str(train_dir), default_transforms(train=True), classes=classes)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    model = LeafClassifier(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    crit = nn.CrossEntropyLoss()
    run_epoch(model, batches(loader, args.warmup, args.mode), crit, opt, device, "warm-up", scaler)
    return timed(lambda: run_epoch(model, batches(loader, args.steps, args.mode), crit, opt, device,
                                   "timed", scaler), device, args.steps * args.batch_size)


def bench_unet(args, amp, device):
    set_seed(42)
    ds = SegmentationDataset(Path(args.data_dir) / "seg" / "images", Path(args.data_dir) / "seg" / "masks", 256)
    loader = DataLoader(ds, batch_size=args.seg_batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True)
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    bce = nn.BCEWithLogitsLoss()

    def steps(n):  # same step as train_segmentation.train()
        model.train()
        for images, masks in batches(loader, n, args.mode):
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad()
            if scaler is None:
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                loss.backward()
                opt.step()
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                logits = logits.float()
                loss = bce(logits, masks) + dice_loss(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

    steps(args.warmup)
    return timed(lambda: steps(args.steps), device, args.steps * args.seg_batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="DATA_DIR used by run_all.sh (prepared data)")
    p.add_argument("--mode", choices=["gpu", "loader"], default="gpu")
    p.add_argument("--batch-size", type=int, default=32, help="classifier (train_classifier default)")
    p.add_argument("--seg-batch-size", type=int, default=8, help="U-Net (train_segmentation default)")
    p.add_argument("--warmup", type=int, default=30, help="untimed batches first (cuDNN autotune, allocator)")
    p.add_argument("--steps", type=int, default=100, help="timed batches")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--json", help="also write the results here")
    args = p.parse_args()

    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"GPU free at start {free / 2**20:.0f}/{total / 2**20:.0f} MiB | mode={args.mode}")
    try:
        util = torch.cuda.utilization()
    except Exception:  # needs pynvml / nvidia-ml-py
        util = None
    if util:
        print(f"WARNING: GPU utilization is {util}% before starting -- another process is using it; "
              f"throughput will be lower and noisier than on an idle GPU.")

    results = {"mode": args.mode, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    for name, fn, bs in [("classifier", bench_classifier, args.batch_size),
                         ("unet", bench_unet, args.seg_batch_size)]:
        for amp in (False, True):
            ips, mem = fn(args, amp, device)
            results[f"{name}_{'amp' if amp else 'fp32'}"] = {"images_per_sec": ips, "peak_mem_mib": mem,
                                                             "batch_size": bs}
            print(f"{name:10s} {'AMP ' if amp else 'fp32'}  batch {bs:3d}: {ips:7.1f} img/s, "
                  f"peak GPU memory {mem:6.0f} MiB")
            torch.cuda.empty_cache()
        f32, a = results[f"{name}_fp32"], results[f"{name}_amp"]
        print(f"{name:10s} AMP speed-up x{a['images_per_sec'] / f32['images_per_sec']:.2f}, "
              f"memory x{a['peak_mem_mib'] / f32['peak_mem_mib']:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
