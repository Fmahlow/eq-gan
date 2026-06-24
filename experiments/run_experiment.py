"""
Main experiment runner for EQ-GAN paper.

Usage:
  python experiments/run_experiment.py --dataset fashionmnist --mode full
  python experiments/run_experiment.py --dataset fashionmnist --mode no_entanglement
  python experiments/run_experiment.py --dataset fashionmnist --mode no_style
  python experiments/run_experiment.py --dataset mnist --mode full
  python experiments/run_experiment.py --dataset breastmnist --mode full
"""

import sys
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.datasets import get_dataset
from src.training.trainer import EQGANTrainer
from src.evaluation.metrics import Evaluator
from src.utils.visualization import (
    plot_generated_grid, plot_loss_curves, plot_fid_curves,
)

FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]
MNIST_CLASSES   = [str(i) for i in range(10)]
BREAST_CLASSES  = ["Benign", "Malignant"]

CLASS_NAMES = {
    "fashionmnist": FASHION_CLASSES,
    "mnist":        MNIST_CLASSES,
    "breastmnist":  BREAST_CLASSES,
}

DATASET_CHANNELS = {
    "fashionmnist": 1,
    "mnist":        1,
    "breastmnist":  1,
}


def subset_loader(full_loader, n_samples: int, batch_size: int):
    """Return a DataLoader over a random n_samples subset."""
    dataset = full_loader.dataset
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)),
                               replace=False)
    sub = Subset(dataset, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="fashionmnist",
                   choices=["fashionmnist", "mnist", "breastmnist"])
    p.add_argument("--mode",       default="full",
                   choices=["full", "no_entanglement", "no_style"])
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--n_train",    type=int,   default=10000,
                   help="Training subset size (QGAN experiments typically 5K-10K)")
    p.add_argument("--lr_g",       type=float, default=1e-4)
    p.add_argument("--lr_d",       type=float, default=1e-4)
    p.add_argument("--noise_dim",  type=int,   default=64)
    p.add_argument("--n_critic",   type=int,   default=3)
    p.add_argument("--lambda_gp",  type=float, default=10.0)
    p.add_argument("--eval_every", type=int,   default=10)
    p.add_argument("--n_eval",     type=int,   default=2000)
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--out_dir",    default="./results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} | Mode: {args.mode} | "
          f"Epochs: {args.epochs} | Train subset: {args.n_train}")

    run_name = f"{args.dataset}_{args.mode}"
    save_dir = os.path.join(args.out_dir, "checkpoints", run_name)
    fig_dir  = os.path.join(args.out_dir, "figures",     run_name)
    log_dir  = os.path.join(args.out_dir, "logs")
    for d in [save_dir, fig_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # Datasets
    full_train_loader, test_loader, n_classes = get_dataset(
        args.dataset, batch_size=args.batch_size, data_root=args.data_root
    )
    train_loader = subset_loader(full_train_loader, args.n_train, args.batch_size)
    print(f"Classes: {n_classes} | Batches/epoch: {len(train_loader)}")

    config = {
        "dataset":      args.dataset,
        "mode":         args.mode,
        "noise_dim":    args.noise_dim,
        "n_classes":    n_classes,
        "img_channels": DATASET_CHANNELS[args.dataset],
        "lr_g":         args.lr_g,
        "lr_d":         args.lr_d,
        "n_critic":     args.n_critic,
        "lambda_gp":    args.lambda_gp,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "n_train":      args.n_train,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Evaluator
    evaluator = Evaluator(device, n_eval_samples=args.n_eval)
    evaluator.cache_real_features(args.dataset, test_loader)

    def eval_fn(generator, epoch):
        metrics = evaluator.evaluate(generator, n_classes, args.dataset)
        with open(os.path.join(save_dir, f"metrics_epoch{epoch:03d}.json"), "w") as f:
            json.dump({**metrics, "epoch": epoch}, f, indent=2)
        print(f"    FID={metrics['fid']:.2f} | "
              f"IS={metrics['is_mean']:.3f}±{metrics['is_std']:.3f} | "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
        plot_generated_grid(
            generator, n_classes, device,
            os.path.join(fig_dir, f"samples_epoch{epoch:03d}.png"),
            class_names=CLASS_NAMES.get(args.dataset),
        )
        return metrics["fid"]

    # Train
    trainer = EQGANTrainer(config, device)
    trainer.fit(
        train_loader,
        n_epochs=args.epochs,
        eval_fn=eval_fn,
        save_dir=save_dir,
        eval_every=args.eval_every,
    )

    # Final figures
    plot_loss_curves(trainer.history,
                     os.path.join(fig_dir, "loss_curves.png"))
    label = "eq_gan" if args.mode == "full" else args.mode
    plot_fid_curves({label: trainer.history},
                    os.path.join(fig_dir, "fid_curve.png"))

    print(f"\nDone. Results in: {save_dir}")


if __name__ == "__main__":
    main()
