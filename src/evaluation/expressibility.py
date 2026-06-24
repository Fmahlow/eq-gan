"""
Expressibility analysis (Sim et al. 2019).
Measures how uniformly a PQC covers the Hilbert space.
"""

import numpy as np
import torch
import pennylane as qml
from scipy.stats import entropy as scipy_entropy

from ..models.quantum_circuit import (
    eq_gan_circuit, ablation_no_entanglement_circuit,
    N_CONTENT, N_STYLE, N_QUBITS, N_LAYERS,
)


def _sample_fidelities(circuit_fn, n_samples: int = 2000, mode: str = "full"):
    """
    Sample pairwise state fidelities |<ψ(θ1)|ψ(θ2)>|^2
    for random parameter pairs. Returns array of fidelities.
    """
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    fidelities = []

    for _ in range(n_samples):
        if mode == "full":
            n_qubits = N_QUBITS
        else:
            n_qubits = N_QUBITS

        params1 = torch.rand(N_LAYERS, n_qubits, 3) * 2 * np.pi
        params2 = torch.rand(N_LAYERS, n_qubits, 3) * 2 * np.pi
        noise1  = torch.rand(N_CONTENT) * 2 * np.pi
        noise2  = torch.rand(N_CONTENT) * 2 * np.pi
        style1  = torch.rand(N_STYLE)  * 2 * np.pi
        style2  = torch.rand(N_STYLE)  * 2 * np.pi

        @qml.qnode(dev)
        def state1():
            if mode == "full":
                eq_gan_circuit.func(noise1, style1, params1)
            else:
                ablation_no_entanglement_circuit.func(noise1, style1, params1)
            return qml.state()

        @qml.qnode(dev)
        def state2():
            if mode == "full":
                eq_gan_circuit.func(noise2, style2, params2)
            else:
                ablation_no_entanglement_circuit.func(noise2, style2, params2)
            return qml.state()

        sv1 = np.array(state1())
        sv2 = np.array(state2())
        fid = np.abs(np.dot(sv1.conj(), sv2)) ** 2
        fidelities.append(float(fid))

    return np.array(fidelities)


def expressibility_kl(fidelities: np.ndarray, n_qubits: int,
                      n_bins: int = 75) -> float:
    """
    KL divergence from the sampled fidelity distribution to the
    Haar-random distribution F(f) = (2^n - 1)(1-f)^(2^n - 2).
    Lower KL → more expressive.
    """
    dim = 2 ** n_qubits
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Empirical distribution
    p_empirical, _ = np.histogram(fidelities, bins=bins, density=True)
    p_empirical = p_empirical / (p_empirical.sum() + 1e-10)

    # Haar-random distribution
    p_haar = (dim - 1) * (1 - bin_centers) ** (dim - 2)
    p_haar = p_haar / (p_haar.sum() + 1e-10)

    kl = scipy_entropy(p_empirical + 1e-10, p_haar + 1e-10)
    return float(kl)


def compute_expressibility(mode: str = "full", n_samples: int = 1000) -> dict:
    """Returns expressibility KL and fidelity statistics for a circuit mode."""
    fidelities = _sample_fidelities(mode=mode, n_samples=n_samples)
    kl = expressibility_kl(fidelities, N_QUBITS)
    return {
        "mode":          mode,
        "kl_divergence": kl,
        "fidelity_mean": float(fidelities.mean()),
        "fidelity_std":  float(fidelities.std()),
        "n_samples":     n_samples,
    }
