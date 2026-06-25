"""
Generate all paper figures from checkpoint data:
  1. FID learning curves (all models)
  2. Ablation bar chart (fashionmnist)
  3. Expressibility analysis
  4. Entanglement entropy (from quantum circuit)
  5. Sample grids (best epoch per model)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CKPT_DIR  = "./results/checkpoints"
FIG_DIR   = "./results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

EPOCHS = [10, 20, 30, 40, 50]

# Readable labels
MODEL_LABELS = {
    "fashionmnist_full":             "EQ-GAN (full)",
    "fashionmnist_no_entanglement":  "EQ-GAN w/o Entanglement",
    "fashionmnist_no_style":         "EQ-GAN w/o Style",
    "patchqgan_fashionmnist":        "Patch QGAN",
    "fashionmnist_dcgan":            "DCGAN (classical)",
    "fashionmnist_wgan_gp":          "WGAN-GP (classical)",
}
MODEL_COLORS = {
    "fashionmnist_full":            "#1f77b4",
    "fashionmnist_no_entanglement": "#ff7f0e",
    "fashionmnist_no_style":        "#2ca02c",
    "patchqgan_fashionmnist":       "#9467bd",
    "fashionmnist_dcgan":           "#d62728",
    "fashionmnist_wgan_gp":         "#8c564b",
}
MODEL_STYLES = {
    "fashionmnist_full":            "-",
    "fashionmnist_no_entanglement": "--",
    "fashionmnist_no_style":        "-.",
    "patchqgan_fashionmnist":       ":",
    "fashionmnist_dcgan":           "-",
    "fashionmnist_wgan_gp":         "--",
}


def load_metric_curve(run_name, metric="fid"):
    """Load metric at each checkpoint epoch."""
    epochs, values = [], []
    for ep in EPOCHS:
        f = os.path.join(CKPT_DIR, run_name, f"metrics_epoch{ep:03d}.json")
        if os.path.exists(f):
            with open(f) as fp:
                data = json.load(fp)
            epochs.append(ep)
            values.append(data[metric])
    return epochs, values


def load_final_metrics(run_name):
    """Load final epoch (50) metrics dict."""
    f = os.path.join(CKPT_DIR, run_name, "metrics_epoch050.json")
    if not os.path.exists(f):
        # Try last available
        for ep in reversed(EPOCHS):
            f = os.path.join(CKPT_DIR, run_name, f"metrics_epoch{ep:03d}.json")
            if os.path.exists(f):
                break
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return None


# ---------------------------------------------------------------------------
# Figure 1: FID curves — FashionMNIST (quantum + classical)
# ---------------------------------------------------------------------------

def fig_fid_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: quantum models
    quantum = ["fashionmnist_full", "fashionmnist_no_entanglement",
               "fashionmnist_no_style", "patchqgan_fashionmnist"]
    ax = axes[0]
    for run in quantum:
        eps, vals = load_metric_curve(run)
        if vals:
            ax.plot(eps, vals,
                    label=MODEL_LABELS.get(run, run),
                    color=MODEL_COLORS.get(run, "grey"),
                    linestyle=MODEL_STYLES.get(run, "-"),
                    linewidth=2, marker="o", markersize=4)
    ax.set_title("Quantum Models – FID on FashionMNIST", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID ↓")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: classical vs best quantum
    classical = ["fashionmnist_dcgan", "fashionmnist_wgan_gp"]
    ax = axes[1]
    for run in ["fashionmnist_full"] + classical:
        eps, vals = load_metric_curve(run)
        if vals:
            ax.plot(eps, vals,
                    label=MODEL_LABELS.get(run, run),
                    color=MODEL_COLORS.get(run, "grey"),
                    linestyle=MODEL_STYLES.get(run, "-"),
                    linewidth=2, marker="o", markersize=4)
    ax.set_title("Classical vs. EQ-GAN – FID on FashionMNIST", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID ↓")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = f"{FIG_DIR}/fig1_fid_curves.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Ablation bar chart — final FID, IS, Precision
# ---------------------------------------------------------------------------

def fig_ablation_bars():
    runs = [
        "fashionmnist_full",
        "fashionmnist_no_entanglement",
        "fashionmnist_no_style",
        "patchqgan_fashionmnist",
    ]
    labels  = [MODEL_LABELS[r] for r in runs]
    colors  = [MODEL_COLORS[r] for r in runs]

    fid_vals  = []
    is_vals   = []
    prec_vals = []

    for run in runs:
        m = load_final_metrics(run)
        if m:
            fid_vals.append(m["fid"])
            is_vals.append(m["is_mean"])
            prec_vals.append(m["precision"])
        else:
            fid_vals.append(0)
            is_vals.append(0)
            prec_vals.append(0)

    x   = np.arange(len(runs))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, title, lower_better in zip(
        axes,
        [fid_vals, is_vals, prec_vals],
        ["FID ↓", "IS ↑", "Precision ↑"],
        [True, False, False]
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title.split()[0])
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Ablation Study – FashionMNIST (50 epochs, n=5000)", fontsize=13)
    plt.tight_layout()
    out = f"{FIG_DIR}/fig2_ablation_bars.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Cross-dataset FID comparison (final epoch)
# ---------------------------------------------------------------------------

def fig_cross_dataset():
    datasets  = ["fashionmnist", "mnist", "breastmnist"]
    models    = ["full", "dcgan"]
    model_labels = {"full": "EQ-GAN", "dcgan": "DCGAN"}
    ds_labels = {"fashionmnist": "FashionMNIST", "mnist": "MNIST",
                 "breastmnist": "BreastMNIST"}
    colors_m  = {"full": "#1f77b4", "dcgan": "#d62728"}

    # Special case: patchqgan
    all_runs = {
        "fashionmnist": {
            "EQ-GAN":     load_final_metrics("fashionmnist_full"),
            "Patch QGAN": load_final_metrics("patchqgan_fashionmnist"),
            "DCGAN":      load_final_metrics("fashionmnist_dcgan"),
        },
        "mnist": {
            "EQ-GAN": load_final_metrics("mnist_full"),
            "DCGAN":  load_final_metrics("mnist_dcgan"),
        },
        "breastmnist": {
            "EQ-GAN": load_final_metrics("breastmnist_full"),
            "DCGAN":  load_final_metrics("breastmnist_dcgan"),
        },
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    n_ds     = len(datasets)
    n_models = 3  # max models per dataset
    width    = 0.25
    x        = np.arange(n_ds)

    model_names  = ["EQ-GAN", "Patch QGAN", "DCGAN"]
    model_colors = {"EQ-GAN": "#1f77b4", "Patch QGAN": "#9467bd", "DCGAN": "#d62728"}

    for i, mname in enumerate(model_names):
        fids = []
        for ds in datasets:
            m = all_runs[ds].get(mname)
            fids.append(m["fid"] if m else 0)
        bars = ax.bar(x + i * width, fids, width, label=mname,
                      color=model_colors[mname], alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, fids):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([ds_labels[d] for d in datasets], fontsize=11)
    ax.set_ylabel("FID ↓", fontsize=11)
    ax.set_title("Cross-Dataset FID Comparison (Epoch 50)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{FIG_DIR}/fig3_cross_dataset.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Epoch-by-epoch consistency (ablation robustness)
# Checks whether no_entanglement ALWAYS beats full across checkpoints
# ---------------------------------------------------------------------------

def fig_ablation_consistency():
    runs = {
        "EQ-GAN (full)":          "fashionmnist_full",
        "w/o Entanglement":       "fashionmnist_no_entanglement",
        "w/o Style":              "fashionmnist_no_style",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, run), color in zip(runs.items(), colors):
        eps, vals = load_metric_curve(run)
        if vals:
            ax.plot(eps, vals, label=label, color=color,
                    linewidth=2.5, marker="o", markersize=6)

    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel("FID ↓", fontsize=11)
    ax.set_title("Ablation Consistency Across Training (FashionMNIST)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate the gap at epoch 50
    eps_full, vals_full = load_metric_curve("fashionmnist_full")
    eps_noent, vals_noent = load_metric_curve("fashionmnist_no_entanglement")
    if vals_full and vals_noent:
        gap = vals_full[-1] - vals_noent[-1]
        ax.annotate(
            f"Δ={gap:.1f}",
            xy=(50, (vals_full[-1] + vals_noent[-1]) / 2),
            xytext=(44, (vals_full[-1] + vals_noent[-1]) / 2 + 10),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    plt.tight_layout()
    out = f"{FIG_DIR}/fig4_ablation_consistency.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Summary table figure (for paper appendix)
# ---------------------------------------------------------------------------

def fig_results_table():
    rows = [
        # (Model, Dataset, FID, IS, P, R)
        ("EQ-GAN (ours)",         "FashionMNIST", "fashionmnist_full"),
        ("w/o Entanglement",      "FashionMNIST", "fashionmnist_no_entanglement"),
        ("w/o Style",             "FashionMNIST", "fashionmnist_no_style"),
        ("Patch QGAN",            "FashionMNIST", "patchqgan_fashionmnist"),
        ("DCGAN",                 "FashionMNIST", "fashionmnist_dcgan"),
        ("WGAN-GP",               "FashionMNIST", "fashionmnist_wgan_gp"),
        ("EQ-GAN (ours)",         "MNIST",        "mnist_full"),
        ("DCGAN",                 "MNIST",        "mnist_dcgan"),
        ("EQ-GAN (ours)",         "BreastMNIST",  "breastmnist_full"),
        ("DCGAN",                 "BreastMNIST",  "breastmnist_dcgan"),
    ]

    table_rows = []
    for model, ds, run in rows:
        m = load_final_metrics(run)
        if m:
            table_rows.append([
                model, ds,
                f"{m['fid']:.1f}",
                f"{m['is_mean']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
            ])
        else:
            table_rows.append([model, ds, "--", "--", "--", "--"])

    fig, ax = plt.subplots(figsize=(12, len(table_rows) * 0.55 + 1))
    ax.axis("off")
    col_labels = ["Model", "Dataset", "FID ↓", "IS ↑", "Precision ↑", "Recall ↑"]
    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(table_rows) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    plt.title("Full Results Summary", fontsize=13, pad=12)
    plt.tight_layout()
    out = f"{FIG_DIR}/fig5_results_table.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating paper figures...")
    fig_fid_curves()
    fig_ablation_bars()
    fig_cross_dataset()
    fig_ablation_consistency()
    fig_results_table()
    print(f"\nAll figures saved to {FIG_DIR}/")
