"""
Expressibility analysis for EQ-GAN circuit variants.
Computes KL divergence from Haar-random distribution for:
  - EQ-GAN full circuit (content+style+entanglement)
  - No-entanglement circuit (content+style, no CNOT bridge)
  - No-style circuit (content only)
  - Patch QGAN circuit (4 qubits, simple HEA)

Also computes entanglement entropy (von Neumann) for EQ-GAN full circuit.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pennylane as qml
import torch

FIG_DIR  = "./results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Expressibility constants
N_SAMPLES  = 2000   # random parameter sets
N_BINS     = 75

# ---------------------------------------------------------------------------
# Circuit definitions (mirrored from quantum_circuit.py for self-containment)
# ---------------------------------------------------------------------------

N_CONTENT = 6
N_STYLE   = 2
N_QUBITS  = 8
N_LAYERS  = 3

dev8 = qml.device("lightning.qubit", wires=N_QUBITS)
dev4 = qml.device("lightning.qubit", wires=4)


@qml.qnode(dev8)
def circuit_full(content_angles, style_angles, var_params):
    # Angle encoding
    for i, w in enumerate(range(N_CONTENT)):
        qml.RY(content_angles[i], wires=w)
    for i, w in enumerate(range(N_CONTENT, N_QUBITS)):
        qml.Hadamard(wires=w)
        qml.RZ(style_angles[i] ** 2, wires=w)
        qml.Hadamard(wires=w)
    # Entanglement core
    content_wires = list(range(N_CONTENT))
    style_wires   = list(range(N_CONTENT, N_QUBITS))
    for i, sw in enumerate(style_wires):
        qml.CNOT(wires=[sw, content_wires[i % N_CONTENT]])
    for i in range(len(style_wires) - 1):
        qml.CNOT(wires=[style_wires[i], style_wires[i + 1]])
    # Variational layers
    for l in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RZ(var_params[l, w, 0], wires=w)
            qml.RY(var_params[l, w, 1], wires=w)
            qml.RZ(var_params[l, w, 2], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.state()


@qml.qnode(dev8)
def circuit_no_entanglement(content_angles, style_angles, var_params):
    for i in range(N_CONTENT):
        qml.RY(content_angles[i], wires=i)
    for i, w in enumerate(range(N_CONTENT, N_QUBITS)):
        qml.Hadamard(wires=w)
        qml.RZ(style_angles[i] ** 2, wires=w)
        qml.Hadamard(wires=w)
    # No entanglement core
    for l in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RZ(var_params[l, w, 0], wires=w)
            qml.RY(var_params[l, w, 1], wires=w)
            qml.RZ(var_params[l, w, 2], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.state()


@qml.qnode(dev8)
def circuit_no_style(content_angles, var_params):
    for i in range(N_CONTENT):
        qml.RY(content_angles[i], wires=i)
    for l in range(N_LAYERS):
        for w in range(N_CONTENT):
            qml.RZ(var_params[l, w, 0], wires=w)
            qml.RY(var_params[l, w, 1], wires=w)
            qml.RZ(var_params[l, w, 2], wires=w)
        for w in range(N_CONTENT - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[N_CONTENT - 1, 0])
    return qml.state()


@qml.qnode(dev4)
def circuit_patch(noise_angles, var_params):
    for i in range(4):
        qml.RY(noise_angles[i], wires=i)
    for l in range(3):
        for w in range(4):
            qml.RZ(var_params[l, w, 0], wires=w)
            qml.RY(var_params[l, w, 1], wires=w)
            qml.RZ(var_params[l, w, 2], wires=w)
        for w in range(3):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[3, 0])
    return qml.state()


# ---------------------------------------------------------------------------
# Expressibility: fidelity histogram vs. Haar
# ---------------------------------------------------------------------------

def fidelity_histogram(circuit_fn, n_qubits, n_samples=N_SAMPLES):
    """Sample random fidelities |<ψ_θ|ψ_φ>|² for the circuit."""
    fidelities = []
    for _ in range(n_samples):
        if n_qubits == 8:
            ca = np.random.uniform(-np.pi, np.pi, N_CONTENT)
            sa = np.random.uniform(-np.pi, np.pi, N_STYLE)
            vp = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 3))
            ca2 = np.random.uniform(-np.pi, np.pi, N_CONTENT)
            sa2 = np.random.uniform(-np.pi, np.pi, N_STYLE)
            vp2 = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 3))
            if circuit_fn == circuit_full:
                s1 = np.array(circuit_fn(ca, sa, vp))
                s2 = np.array(circuit_fn(ca2, sa2, vp2))
            elif circuit_fn == circuit_no_entanglement:
                s1 = np.array(circuit_fn(ca, sa, vp))
                s2 = np.array(circuit_fn(ca2, sa2, vp2))
            else:  # no_style
                vp_ns  = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_CONTENT, 3))
                vp_ns2 = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_CONTENT, 3))
                s1 = np.array(circuit_fn(ca, vp_ns))
                s2 = np.array(circuit_fn(ca2, vp_ns2))
        else:  # 4-qubit patch
            na  = np.random.uniform(-np.pi, np.pi, 4)
            vp  = np.random.uniform(-np.pi, np.pi, (3, 4, 3))
            na2 = np.random.uniform(-np.pi, np.pi, 4)
            vp2 = np.random.uniform(-np.pi, np.pi, (3, 4, 3))
            s1  = np.array(circuit_fn(na, vp))
            s2  = np.array(circuit_fn(na2, vp2))

        fidelities.append(abs(np.dot(s1.conj(), s2)) ** 2)
    return np.array(fidelities, dtype=float)


def haar_cdf(f, n_qubits):
    """Theoretical Haar CDF: P(F ≤ f) = 1 - (1-f)^(d-1) where d=2^n."""
    d = 2 ** n_qubits
    return 1 - (1 - np.array(f)) ** (d - 1)


def expressibility_kl(fidelities, n_qubits, n_bins=N_BINS):
    """KL divergence between circuit fidelity histogram and Haar distribution."""
    bins  = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(fidelities, bins=bins, density=True)
    bin_w   = 1 / n_bins

    # Haar pdf: d=2^n, P(f) = (d-1)(1-f)^(d-2)
    d = 2 ** n_qubits
    bin_centers = (bins[:-1] + bins[1:]) / 2
    haar_pdf = (d - 1) * (1 - bin_centers) ** (d - 2)
    haar_pdf = haar_pdf / haar_pdf.sum()
    hist_n   = hist * bin_w
    hist_n   = hist_n / hist_n.sum()

    kl = np.sum(hist_n * np.log((hist_n + 1e-10) / (haar_pdf + 1e-10)))
    return float(kl)


# ---------------------------------------------------------------------------
# Entanglement entropy via von Neumann
# ---------------------------------------------------------------------------

def von_neumann_entropy(state_vector, subsystem_wires, total_wires):
    """Compute S(ρ_A) for a subsystem of a pure state."""
    n = total_wires
    dim = 2 ** n
    psi = np.array(state_vector).reshape([2] * n)

    # Reshape to (dim_A, dim_B) where A = subsystem_wires
    other_wires = [w for w in range(n) if w not in subsystem_wires]
    order       = subsystem_wires + other_wires
    psi_reorder = psi.transpose(order).reshape(
        2 ** len(subsystem_wires), 2 ** len(other_wires)
    )

    singular_values = np.linalg.svd(psi_reorder, compute_uv=False)
    probs = singular_values ** 2
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log2(probs)))


def sample_entanglement_entropy(n_samples=500):
    """
    Sample average entanglement entropy between content and style registers
    for the full EQ-GAN circuit across random parameters.
    """
    entropies_before, entropies_after = [], []
    style_wires = list(range(N_CONTENT, N_QUBITS))

    for _ in range(n_samples):
        ca = np.random.uniform(-np.pi, np.pi, N_CONTENT)
        sa = np.random.uniform(-np.pi, np.pi, N_STYLE)
        vp = np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 3))
        vp_zero = np.zeros((N_LAYERS, N_QUBITS, 3))  # no variational — just encoding

        s_before = np.array(circuit_no_entanglement(ca, sa, vp_zero))
        s_after  = np.array(circuit_full(ca, sa, vp))

        entropies_before.append(von_neumann_entropy(s_before, style_wires, N_QUBITS))
        entropies_after.append( von_neumann_entropy(s_after,  style_wires, N_QUBITS))

    return np.array(entropies_before), np.array(entropies_after)


# ---------------------------------------------------------------------------
# Main: run all analyses and generate figures
# ---------------------------------------------------------------------------

def main():
    print("Computing expressibility (fidelity histograms)...")
    print("  - EQ-GAN full...")
    fid_full = fidelity_histogram(circuit_full,           N_QUBITS, n_samples=N_SAMPLES)
    print(f"    KL = {expressibility_kl(fid_full, N_QUBITS):.4f}")

    print("  - No entanglement...")
    fid_noent = fidelity_histogram(circuit_no_entanglement, N_QUBITS, n_samples=N_SAMPLES)
    print(f"    KL = {expressibility_kl(fid_noent, N_QUBITS):.4f}")

    print("  - No style...")
    fid_nostyle = fidelity_histogram(circuit_no_style,    N_QUBITS, n_samples=N_SAMPLES)
    print(f"    KL = {expressibility_kl(fid_nostyle, N_QUBITS):.4f}")

    print("  - Patch QGAN (4 qubits)...")
    fid_patch = fidelity_histogram(circuit_patch,          4,        n_samples=N_SAMPLES)
    print(f"    KL = {expressibility_kl(fid_patch, 4):.4f}")

    # ---- Figure 6: Expressibility plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(0, 1, N_BINS + 1)
    x    = (bins[:-1] + bins[1:]) / 2

    # Left: fidelity distributions vs Haar (8-qubit)
    ax = axes[0]
    ax.set_title("Fidelity Distribution vs. Haar (8 qubits)", fontsize=11)
    haar_8 = (2**N_QUBITS - 1) * (1 - x) ** (2**N_QUBITS - 2)

    for fids, label, color, ls in [
        (fid_full,    "EQ-GAN (full)",          "#1f77b4", "-"),
        (fid_noent,   "w/o Entanglement",        "#ff7f0e", "--"),
        (fid_nostyle, "w/o Style (6 qubits)",    "#2ca02c", "-."),
    ]:
        h, _ = np.histogram(fids, bins=bins, density=True)
        ax.plot(x, h, label=label, color=color, linestyle=ls, linewidth=1.8)
    ax.plot(x, haar_8, "k--", linewidth=1.5, label="Haar (8 qubits)", alpha=0.6)
    ax.set_xlabel("Fidelity |⟨ψ_θ|ψ_φ⟩|²")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.05)  # zoom in — 8-qubit Haar concentrates near 0

    # Right: KL divergence summary bar
    ax = axes[1]
    kl_vals = [
        expressibility_kl(fid_full,    N_QUBITS),
        expressibility_kl(fid_noent,   N_QUBITS),
        expressibility_kl(fid_nostyle, N_QUBITS),
        expressibility_kl(fid_patch,   4),
    ]
    bar_labels = ["EQ-GAN\n(full)", "w/o\nEntanglement", "w/o\nStyle", "Patch\nQGAN\n(4q)"]
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = ax.bar(bar_labels, kl_vals, color=bar_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Expressibility (KL divergence) ↓", fontsize=10)
    ax.set_title("Circuit Expressibility\n(lower = more expressive)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = f"{FIG_DIR}/fig6_expressibility.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ---- Figure 7: Entanglement entropy ----
    print("\nComputing entanglement entropy (von Neumann)...")
    ent_before, ent_after = sample_entanglement_entropy(n_samples=500)
    print(f"  Before entanglement core: {ent_before.mean():.4f} ± {ent_before.std():.4f}")
    print(f"  After  full circuit:      {ent_after.mean():.4f} ± {ent_after.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(ent_before, bins=40, alpha=0.7, color="#ff7f0e",
            label="Encoding only (no entanglement core)", edgecolor="white")
    ax.hist(ent_after,  bins=40, alpha=0.7, color="#1f77b4",
            label="Full EQ-GAN circuit", edgecolor="white")
    ax.axvline(ent_before.mean(), color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.axvline(ent_after.mean(),  color="#1f77b4",  linestyle="--", linewidth=1.5)
    ax.set_xlabel("Von Neumann Entropy S(ρ_style) [bits]", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Entanglement Entropy Distribution\n(Style Register, 500 random params)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(["w/o Entanglement\nCore", "Full EQ-GAN\nCircuit"],
           [ent_before.mean(), ent_after.mean()],
           yerr=[ent_before.std(), ent_after.std()],
           color=["#ff7f0e", "#1f77b4"], alpha=0.85,
           edgecolor="black", capsize=6, linewidth=0.5)
    ax.set_ylabel("Mean S(ρ_style) [bits]", fontsize=10)
    ax.set_title("Mean Entanglement Entropy\n(Style ↔ Content Register)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Theoretical max for 2-qubit system = 1 bit (Bell state)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.2, label="Max (Bell state)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"{FIG_DIR}/fig7_entanglement_entropy.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ---- Save KL values to JSON ----
    results = {
        "eq_gan_full":          expressibility_kl(fid_full,    N_QUBITS),
        "no_entanglement":      expressibility_kl(fid_noent,   N_QUBITS),
        "no_style":             expressibility_kl(fid_nostyle, N_QUBITS),
        "patch_qgan_4q":        expressibility_kl(fid_patch,   4),
        "entropy_no_core_mean": float(ent_before.mean()),
        "entropy_no_core_std":  float(ent_before.std()),
        "entropy_full_mean":    float(ent_after.mean()),
        "entropy_full_std":     float(ent_after.std()),
    }
    import json
    with open(f"{FIG_DIR}/expressibility_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExpressibility + entropy results saved to {FIG_DIR}/expressibility_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
