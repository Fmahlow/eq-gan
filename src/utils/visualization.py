"""Publication-quality figures for EQ-GAN paper."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


PALETTE = {
    "eq_gan":              "#1f77b4",
    "no_entanglement":     "#ff7f0e",
    "no_style":            "#2ca02c",
    "dcgan":               "#9467bd",
    "wgan_gp":             "#8c564b",
    "patch_qgan":          "#e377c2",
    "mosaiq":              "#7f7f7f",
    "latent_qgan":         "#bcbd22",
}

FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE   = (10, 4)
DPI            = 300


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_generated_grid(generator, n_classes: int, device: torch.device,
                        save_path: str, n_per_class: int = 8,
                        class_names: list = None):
    """Grid: rows=classes, cols=samples."""
    generator.eval()
    import torch.nn.functional as F

    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 0.8, n_classes * 0.8))
    with torch.no_grad():
        for c in range(n_classes):
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            imgs = generator.sample(n_per_class, labels, device).cpu()
            imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
            for j in range(n_per_class):
                ax = axes[c, j] if n_classes > 1 else axes[j]
                img = imgs[j].squeeze().numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if j == 0 and class_names:
                    ax.set_ylabel(class_names[c], fontsize=7, rotation=90,
                                  labelpad=2)

    plt.suptitle("EQ-GAN Generated Samples", y=1.01, fontsize=9)
    plt.tight_layout()
    _savefig(fig, save_path)
    generator.train()


def plot_image_grid(imgs: "torch.Tensor", save_path: str, title: str = None):
    """Save a (N, C, H, W) or (N, H, W) tensor as a single-row image grid."""
    import torch
    imgs = imgs.float()
    if imgs.min() < 0:
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
    imgs = imgs.clamp(0, 1)
    n = imgs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(n * 0.8, 0.9))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        plt.suptitle(title, y=1.01, fontsize=8)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_fid_curves(histories: dict, save_path: str):
    """FID over training epochs for multiple models."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for name, history in histories.items():
        epochs = [e for e, _ in history.get("fid", [])]
        fids   = [f for _, f in history.get("fid", [])]
        if epochs:
            ax.plot(epochs, fids, label=name, color=PALETTE.get(name, None),
                    linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID During Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_loss_curves(history: dict, save_path: str):
    """Generator and discriminator loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    epochs = range(1, len(history["d_loss"]) + 1)

    ax1.plot(epochs, history["d_loss"], color="#e41a1c", linewidth=1)
    ax1.set_title("Discriminator Loss (WGAN)")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["g_loss"], color="#377eb8", linewidth=1)
    ax2.set_title("Generator Loss (WGAN)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("EQ-GAN Training Curves", fontsize=10)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_results_table_figure(results: dict, save_path: str):
    """
    Bar chart comparing FID across models and datasets.
    results: {model_name: {dataset: {"fid": float, ...}}}
    """
    datasets = list(next(iter(results.values())).keys())
    models   = list(results.keys())
    n_ds = len(datasets)
    x = np.arange(len(models))
    width = 0.8 / n_ds

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4393c3", "#f4a582", "#92c5de"]
    for i, ds in enumerate(datasets):
        fids = [results[m].get(ds, {}).get("fid", np.nan) for m in models]
        ax.bar(x + i * width, fids, width, label=ds, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (n_ds - 1) / 2)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID Comparison Across Models and Datasets")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_entanglement_entropy(entropy_log: list, save_path: str):
    """Plot entanglement entropy across training epochs."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    epochs  = [e for e, _ in entropy_log]
    entropies = [v for _, v in entropy_log]

    ax.plot(epochs, entropies, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Von Neumann Entanglement Entropy (bits)")
    ax.set_title("Entanglement Entropy During Training\n(content–style partition)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_ablation_bar(ablation_results: dict, metric: str, save_path: str):
    """Bar chart comparing ablation variants on a single metric."""
    labels = list(ablation_results.keys())
    values = [ablation_results[k].get(metric, np.nan) for k in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [PALETTE.get(k.lower().replace(" ", "_"), "#1f77b4") for k in labels]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ylabel = "FID (↓)" if metric == "fid" else metric
    ax.set_ylabel(ylabel)
    ax.set_title(f"Ablation Study — {metric.upper()}")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_expressibility(expr_results: list, save_path: str):
    """Bar chart of KL divergence (expressibility) for circuit variants."""
    names = [r["mode"] for r in expr_results]
    kls   = [r["kl_divergence"] for r in expr_results]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [PALETTE.get(n, "#1f77b4") for n in names]
    bars = ax.bar(names, kls, color=colors, alpha=0.85, width=0.4)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_ylabel("KL Divergence from Haar (↓ = more expressive)")
    ax.set_title("Circuit Expressibility")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)
