"""
Classical baseline trainer (DCGAN / WGAN-GP).
Runs fully on GPU. Much faster than quantum variants.

Usage:
  python experiments/run_classical_baseline.py --dataset fashionmnist --model dcgan
  python experiments/run_classical_baseline.py --dataset fashionmnist --model wgan_gp
"""

import sys
import os
import json
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.datasets import get_dataset
from src.models.classical_baselines import (
    DCGANGenerator, DCGANDiscriminator, gradient_penalty_classical,
)
from src.evaluation.metrics import Evaluator
from src.utils.visualization import plot_generated_grid, plot_loss_curves

FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]
CLASS_NAMES = {
    "fashionmnist": FASHION_CLASSES,
    "mnist":        [str(i) for i in range(10)],
    "breastmnist":  ["Benign", "Malignant"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="fashionmnist",
                   choices=["fashionmnist", "mnist", "breastmnist"])
    p.add_argument("--model",      default="wgan_gp",
                   choices=["dcgan", "wgan_gp"])
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--n_train",    type=int,   default=5000)
    p.add_argument("--lr_g",       type=float, default=2e-4)
    p.add_argument("--lr_d",       type=float, default=2e-4)
    p.add_argument("--noise_dim",  type=int,   default=64)
    p.add_argument("--n_critic",   type=int,   default=3)
    p.add_argument("--lambda_gp",  type=float, default=10.0)
    p.add_argument("--eval_every", type=int,   default=10)
    p.add_argument("--n_eval",     type=int,   default=1000)
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--out_dir",    default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset} | Model: {args.model}")

    run_name = f"{args.dataset}_{args.model}"
    save_dir = os.path.join(args.out_dir, "checkpoints", run_name)
    fig_dir  = os.path.join(args.out_dir, "figures",     run_name)
    log_dir  = os.path.join(args.out_dir, "logs")
    for d in [save_dir, fig_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    full_train_loader, test_loader, n_classes = get_dataset(
        args.dataset, batch_size=args.batch_size, data_root=args.data_root
    )
    indices = np.random.choice(len(full_train_loader.dataset),
                               min(args.n_train, len(full_train_loader.dataset)),
                               replace=False)
    train_loader = DataLoader(
        Subset(full_train_loader.dataset, indices),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    print(f"Classes: {n_classes} | Batches/epoch: {len(train_loader)}")

    gen  = DCGANGenerator(args.noise_dim, n_classes).to(device)
    disc = DCGANDiscriminator(n_classes=n_classes).to(device)

    betas = (0.5, 0.9) if args.model == "dcgan" else (0.0, 0.9)
    opt_g = torch.optim.Adam(gen.parameters(),  lr=args.lr_g, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=betas)

    evaluator = Evaluator(device, n_eval_samples=args.n_eval)
    evaluator.cache_real_features(args.dataset, test_loader)

    history = {"d_loss": [], "g_loss": [], "fid": [], "epoch_time": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        d_losses, g_losses = [], []

        for real_imgs, real_labels in train_loader:
            real_imgs   = real_imgs.to(device)
            real_labels = real_labels.to(device)
            B = real_imgs.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(B, args.noise_dim, device=device)
                fake_labels = torch.randint(0, n_classes, (B,), device=device)
                with torch.no_grad():
                    fake = gen(z, fake_labels)
                d_real = disc(real_imgs, real_labels)
                d_fake = disc(fake, fake_labels)

                if args.model == "wgan_gp":
                    gp = gradient_penalty_classical(
                        disc, real_imgs, fake, real_labels, device
                    )
                    d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
                else:
                    d_loss = (F.binary_cross_entropy_with_logits(
                        d_real, torch.ones_like(d_real)) +
                        F.binary_cross_entropy_with_logits(
                        d_fake, torch.zeros_like(d_fake)))

                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                d_losses.append(d_loss.item())

            z = torch.randn(B, args.noise_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (B,), device=device)
            fake = gen(z, fake_labels)
            g_out = disc(fake, fake_labels)

            if args.model == "wgan_gp":
                g_loss = -g_out.mean()
            else:
                g_loss = F.binary_cross_entropy_with_logits(
                    g_out, torch.ones_like(g_out))

            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            g_losses.append(g_loss.item())

        elapsed = time.time() - t0
        history["d_loss"].append(np.mean(d_losses))
        history["g_loss"].append(np.mean(g_losses))
        history["epoch_time"].append(elapsed)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"D: {np.mean(d_losses):+.4f} | G: {np.mean(g_losses):+.4f} | "
              f"Time: {elapsed:.1f}s")

        if epoch % args.eval_every == 0:
            metrics = evaluator.evaluate(gen, n_classes, args.dataset)
            history["fid"].append((epoch, metrics["fid"]))
            print(f"  → FID={metrics['fid']:.2f} | "
                  f"IS={metrics['is_mean']:.3f} | "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
            with open(os.path.join(save_dir, f"metrics_epoch{epoch:03d}.json"), "w") as f:
                json.dump({**metrics, "epoch": epoch}, f, indent=2)
            plot_generated_grid(
                gen, n_classes, device,
                os.path.join(fig_dir, f"samples_epoch{epoch:03d}.png"),
                class_names=CLASS_NAMES.get(args.dataset),
            )

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(gen.state_dict(), os.path.join(save_dir, "generator_final.pt"))
    print(f"\nDone. Results in: {save_dir}")


if __name__ == "__main__":
    main()
