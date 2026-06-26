"""
Patch QGAN baseline — Huang et al. (2021) style.

Architecture:
  4 qubits per sub-circuit, N_PATCHES=4 sub-generators.
  Each sub-generator produces a 14×14 image patch.
  Patches are stitched into a 28×28 image by a classical assembler.
  Conditional via classical label embedding concatenated to noise angles.
  Training: WGAN-GP (same as EQ-GAN for fair comparison).
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from .parallel_quantum import OptimizedPatchQuantumLayer

N_PATCH_QUBITS = 4
N_PATCH_LAYERS = 3
N_PATCHES = 4          # 4 sub-generators → 2×2 grid of 14×14 patches

_patch_dev = qml.device("lightning.qubit", wires=N_PATCH_QUBITS)


@qml.qnode(_patch_dev, interface="torch", diff_method="adjoint")
def patch_circuit(noise_inputs, var_params):
    """
    Single patch sub-circuit: angle encoding + HEA, no style register.
    noise_inputs: (N_PATCH_QUBITS,) — includes class info via classical pre-mixing
    var_params:   (N_PATCH_LAYERS, N_PATCH_QUBITS, 3)
    Returns: [N_PATCH_QUBITS Pauli-Z expectation values]
    """
    for i in range(N_PATCH_QUBITS):
        qml.RY(noise_inputs[i], wires=i)
    for l in range(N_PATCH_LAYERS):
        for w in range(N_PATCH_QUBITS):
            qml.RZ(var_params[l, w, 0], wires=w)
            qml.RY(var_params[l, w, 1], wires=w)
            qml.RZ(var_params[l, w, 2], wires=w)
        for w in range(N_PATCH_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[N_PATCH_QUBITS - 1, 0])
    return [qml.expval(qml.PauliZ(w)) for w in range(N_PATCH_QUBITS)]


class PatchQuantumLayer(nn.Module):
    """Runs N_PATCHES sub-circuits, each with independent variational params."""
    def __init__(self):
        super().__init__()
        # Independent params per patch
        self.var_params = nn.ParameterList([
            nn.Parameter(torch.randn(N_PATCH_LAYERS, N_PATCH_QUBITS, 3) * 0.1)
            for _ in range(N_PATCHES)
        ])

    def forward(self, noise_enc):
        """
        noise_enc: (B, N_PATCHES, N_PATCH_QUBITS)
        Returns:   (B, N_PATCHES, N_PATCH_QUBITS)
        """
        noise_cpu = noise_enc.cpu()
        B = noise_cpu.shape[0]
        patch_outputs = []

        for p in range(N_PATCHES):
            params_cpu = self.var_params[p].cpu()
            samples = []
            for i in range(B):
                out = patch_circuit(noise_cpu[i, p], params_cpu)
                samples.append(torch.stack(out))
            patch_outputs.append(torch.stack(samples))  # (B, N_PATCH_QUBITS)

        result = torch.stack(patch_outputs, dim=1)  # (B, N_PATCHES, N_PATCH_QUBITS)
        return result.float().to(noise_enc.device)


class PatchNoiseEncoder(nn.Module):
    """
    Maps noise z + class label → per-patch angle inputs.
    Classical mixing of label info into noise angles (conditional without style register).
    """
    def __init__(self, noise_dim: int = 32, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + n_classes, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, N_PATCHES * N_PATCH_QUBITS),
            nn.Tanh(),
        )
        self.noise_dim = noise_dim

    def forward(self, z, labels_onehot):
        x = torch.cat([z, labels_onehot], dim=1)
        out = self.net(x) * np.pi  # → angles in [-π, π]
        return out.view(-1, N_PATCHES, N_PATCH_QUBITS)


class PatchAssembler(nn.Module):
    """
    Assembles N_PATCHES × N_PATCH_QUBITS quantum outputs → 28×28 image.
    Each patch decoded independently then stitched 2×2.
    """
    def __init__(self, img_channels: int = 1):
        super().__init__()
        # Per-patch decoder: 4 values → 14×14
        self.patch_decoder = nn.Sequential(
            nn.Linear(N_PATCH_QUBITS, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            # 4×4 → 7×7
            nn.ConvTranspose2d(256, 128, 4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, patch_features):
        """
        patch_features: (B, N_PATCHES, N_PATCH_QUBITS)
        Returns: (B, C, 28, 28) — 2×2 grid of 14×14 patches
        """
        B = patch_features.shape[0]
        patches = []
        for p in range(N_PATCHES):
            patch = self.patch_decoder(patch_features[:, p, :])  # (B, C, 14, 14)
            patches.append(patch)

        # Stitch 2×2: [P0 P1 / P2 P3]
        top    = torch.cat([patches[0], patches[1]], dim=3)  # (B, C, 14, 28)
        bottom = torch.cat([patches[2], patches[3]], dim=3)
        return torch.cat([top, bottom], dim=2)               # (B, C, 28, 28)


class PatchQGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 32, n_classes: int = 10,
                 img_channels: int = 1):
        super().__init__()
        self.noise_dim  = noise_dim
        self.n_classes  = n_classes
        self.encoder    = PatchNoiseEncoder(noise_dim, n_classes)
        self.quantum    = OptimizedPatchQuantumLayer()
        self.assembler  = PatchAssembler(img_channels)

    def forward(self, z, labels_onehot):
        noise_enc   = self.encoder(z, labels_onehot)      # (B, N_PATCHES, 4)
        quantum_out = self.quantum(noise_enc)              # (B, N_PATCHES, 4)
        return self.assembler(quantum_out)                 # (B, C, 28, 28)

    def sample(self, n: int, labels: torch.Tensor, device: torch.device):
        import torch.nn.functional as F
        z = torch.randn(n, self.noise_dim, device=device)
        labels_oh = F.one_hot(labels, self.n_classes).float()
        return self(z, labels_oh)
