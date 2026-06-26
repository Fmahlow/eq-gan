"""
Patch QGAN experiment runner — Huang et al. (2021) style quantum baseline.
Usage:
  python3 experiments/run_patch_qgan.py --dataset fashionmnist --epochs 50
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.patch_qgan import PatchQGANGenerator
from src.models.classical_baselines import DCGANDiscriminator, gradient_penalty_classical
from src.data.datasets import get_dataset
from src.evaluation.metrics import Evaluator
from src.utils.visualization import plot_image_grid, plot_fid_curves

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subset_loader(loader, n_samples: int, batch_size: int):
    imgs_all, labels_all = [], []
    for imgs, labels in loader:
        imgs_all.append(imgs)
        labels_all.append(labels)
        if sum(x.shape[0] for x in imgs_all) >= n_samples:
            break
    imgs_all   = torch.cat(imgs_all)[:n_samples]
    labels_all = torch.cat(labels_all)[:n_samples]
    ds = torch.utils.data.TensorDataset(imgs_all, labels_all)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def train(args):
    os.makedirs(f"{args.out_dir}/checkpoints/patchqgan_{args.dataset}", exist_ok=True)
    os.makedirs(f"{args.out_dir}/logs",   exist_ok=True)
    os.makedirs(f"{args.out_dir}/images", exist_ok=True)

    train_loader_full, _, n_classes = get_dataset(
        args.dataset, batch_size=args.batch_size, data_root=args.data_root
    )
    img_channels = 1  # all datasets are grayscale

    train_loader = subset_loader(train_loader_full, args.n_train, args.batch_size)
    _, test_loader, _ = get_dataset(
        args.dataset, batch_size=64, data_root=args.data_root
    )

    generator     = PatchQGANGenerator(
        noise_dim=32, n_classes=n_classes, img_channels=img_channels
    ).to(DEVICE)
    discriminator = DCGANDiscriminator(
        img_channels=img_channels, n_classes=n_classes, img_size=28
    ).to(DEVICE)

    opt_g = optim.Adam(generator.parameters(),     lr=1e-4, betas=(0.0, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    evaluator = Evaluator(DEVICE, n_eval_samples=args.n_eval)
    evaluator.cache_real_features(args.dataset, test_loader)

    history = {"epochs": [], "d_loss": [], "g_loss": [], "fid": [], "is_mean": []}

    print(f"Dataset: {args.dataset} | Patch QGAN | Epochs: {args.epochs} | "
          f"Train subset: {args.n_train}")
    print(f"Classes: {n_classes} | Batches/epoch: {len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        d_losses, g_losses = [], []

        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            B      = imgs.size(0)
            labels_oh = F.one_hot(labels, n_classes).float()

            # -- Discriminator step --
            for _ in range(args.n_critic):
                z    = torch.randn(B, 32, device=DEVICE)
                fake = generator(z, labels_oh).detach()
                gp   = gradient_penalty_classical(discriminator, imgs, fake,
                                                  labels, DEVICE)
                d_real = discriminator(imgs, labels).mean()
                d_fake = discriminator(fake, labels).mean()
                d_loss = d_fake - d_real + args.lambda_gp * gp

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_losses.append(d_loss.item())

            # -- Generator step --
            z    = torch.randn(B, 32, device=DEVICE)
            fake = generator(z, labels_oh)
            g_loss = -discriminator(fake, labels).mean()

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())

        elapsed = time.time() - t0
        d_mean  = sum(d_losses) / len(d_losses)
        g_mean  = sum(g_losses) / len(g_losses)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"D: {d_mean:+.4f} | G: {g_mean:+.4f} | Time: {elapsed:.1f}s")

        history["epochs"].append(epoch)
        history["d_loss"].append(d_mean)
        history["g_loss"].append(g_mean)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluator.evaluate(generator, n_classes, args.dataset)
            fid, is_m = metrics["fid"], metrics["is_mean"]
            print(f"      FID={fid:.2f} | IS={is_m:.3f}±{metrics['is_std']:.3f} | "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

            history["fid"].append(fid)
            history["is_mean"].append(is_m)

            tag    = f"patchqgan_{args.dataset}"
            ckpt   = f"{args.out_dir}/checkpoints/{tag}/checkpoint_epoch{epoch:03d}.pt"
            met_f  = f"{args.out_dir}/checkpoints/{tag}/metrics_epoch{epoch:03d}.json"
            torch.save({"generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict()}, ckpt)
            with open(met_f, "w") as f:
                json.dump({**metrics, "epoch": epoch}, f, indent=2)

            # sample grid
            generator.eval()
            with torch.no_grad():
                sample_labels = torch.arange(n_classes, device=DEVICE).repeat(
                    max(1, 8 // n_classes) + 1
                )[:8]
                sample_imgs = generator.sample(8, sample_labels, DEVICE)
            generator.train()
            plot_image_grid(
                sample_imgs.cpu(),
                f"{args.out_dir}/images/{tag}_epoch{epoch:03d}.png",
                title=f"Patch QGAN – {args.dataset} – epoch {epoch}"
            )

    with open(f"{args.out_dir}/checkpoints/patchqgan_{args.dataset}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Results in: {args.out_dir}/checkpoints/patchqgan_{args.dataset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="fashionmnist",
                   choices=["mnist", "fashionmnist", "breastmnist"])
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_train",    type=int, default=5000)
    p.add_argument("--n_critic",   type=int, default=3)
    p.add_argument("--lambda_gp",  type=float, default=10.0)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--n_eval",     type=int, default=1000)
    p.add_argument("--out_dir",    default="./results")
    p.add_argument("--data_root",  default="./data")
    train(p.parse_args())


if __name__ == "__main__":
    main()
