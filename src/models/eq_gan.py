"""
EQ-GAN: Entanglement-Enhanced Quantum GAN with Latent Style Control.

Generator:
  Classical Encoder → quantum circuit (content+style+entanglement) → Classical Decoder

Discriminator:
  Standard CNN (PatchGAN-style)

Training objective: WGAN-GP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .quantum_circuit import (
    eq_gan_circuit,
    ablation_no_entanglement_circuit,
    ablation_no_style_circuit,
    N_CONTENT, N_STYLE, N_QUBITS, N_LAYERS,
)
from .parallel_quantum import OptimizedEQGANQuantumLayer


# ---------------------------------------------------------------------------
# Generator components
# ---------------------------------------------------------------------------

class ClassicalEncoder(nn.Module):
    """Maps noise z ~ N(0,1) to angle-encoding inputs for content register."""
    def __init__(self, noise_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, N_CONTENT),
            nn.Tanh(),  # outputs in [-1, 1] → angles in [-π, π] after scaling
        )
        self.noise_dim = noise_dim

    def forward(self, z):
        return self.net(z) * np.pi  # scale to [-π, π]


class StyleEncoder(nn.Module):
    """Maps one-hot class label to IQP feature map inputs for style register."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, N_STYLE),
            nn.Tanh(),
        )

    def forward(self, labels_onehot):
        return self.net(labels_onehot) * np.pi


class QuantumLayer(nn.Module):
    """
    Wraps the PennyLane quantum circuit as a PyTorch module.
    Variational parameters are trainable.

    mode: 'full' | 'no_entanglement' | 'no_style'
    """
    def __init__(self, mode: str = "full"):
        super().__init__()
        assert mode in ("full", "no_entanglement", "no_style")
        self.mode = mode

        # Variational parameters: (layers, qubits, 3 rotation angles)
        n_qubits = N_CONTENT if mode == "no_style" else N_QUBITS
        self.var_params = nn.Parameter(
            torch.randn(N_LAYERS, n_qubits, 3) * 0.1
        )

    def forward(self, noise_enc, style_enc=None):
        """
        Args:
            noise_enc: (B, N_CONTENT) — encoded noise (GPU)
            style_enc: (B, N_STYLE)   — encoded style (GPU, None for no_style)
        Returns:
            (B, N_CONTENT) quantum expectation values (GPU)
        """
        # Move all inputs to CPU once before the loop.
        # var_params may be on GPU (moved by .to(device)); .cpu() keeps grad graph intact.
        noise_cpu  = noise_enc.cpu()
        style_cpu  = style_enc.cpu() if style_enc is not None else None
        params_cpu = self.var_params.cpu()

        batch_size = noise_cpu.shape[0]
        outputs = []

        for i in range(batch_size):
            if self.mode == "full":
                out = eq_gan_circuit(noise_cpu[i], style_cpu[i], params_cpu)
            elif self.mode == "no_entanglement":
                out = ablation_no_entanglement_circuit(
                    noise_cpu[i], style_cpu[i], params_cpu
                )
            else:  # no_style
                out = ablation_no_style_circuit(noise_cpu[i], params_cpu)
            outputs.append(torch.stack(out))

        result = torch.stack(outputs)  # (B, N_CONTENT) on CPU
        return result.float().to(noise_enc.device)  # cast to float32 and move back to GPU for decoder


class ClassicalDecoder(nn.Module):
    """
    Maps quantum output (N_CONTENT values) → 28×28 image.
    Uses transposed convolutions for spatial upsampling.
    4×4 → 7×7 → 14×14 → 28×28
    """
    def __init__(self, img_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(N_CONTENT, 512 * 4 * 4)
        self.net = nn.Sequential(
            # 4×4 → 7×7  (kernel=4, stride=1, pad=0: out=(4-1)*1+4=7)
            nn.ConvTranspose2d(512, 256, 4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7×7 → 14×14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14×14 → 28×28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 4, 4)
        return self.net(x)


class EQGANGenerator(nn.Module):
    """
    Full EQ-GAN Generator:
      z, label → ClassicalEncoder → QuantumLayer → ClassicalDecoder → image
    """
    def __init__(self, noise_dim: int = 64, n_classes: int = 10,
                 img_channels: int = 1, mode: str = "full"):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.mode = mode

        self.content_encoder = ClassicalEncoder(noise_dim)
        self.style_encoder = StyleEncoder(n_classes) if mode != "no_style" else None
        self.quantum_layer = OptimizedEQGANQuantumLayer(mode=mode)
        self.decoder = ClassicalDecoder(img_channels)

    def forward(self, z, labels_onehot=None):
        noise_enc = self.content_encoder(z)
        style_enc = self.style_encoder(labels_onehot) if self.style_encoder else None
        quantum_out = self.quantum_layer(noise_enc, style_enc)
        return self.decoder(quantum_out)

    def sample(self, n_samples: int, labels: torch.Tensor, device: torch.device):
        z = torch.randn(n_samples, self.noise_dim, device=device)
        labels_onehot = F.one_hot(labels, self.n_classes).float()
        return self(z, labels_onehot)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class EQGANDiscriminator(nn.Module):
    """
    Conditional CNN discriminator.
    Concatenates class embedding (projected to spatial map) with image.
    """
    def __init__(self, img_channels: int = 1, n_classes: int = 10, img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.label_embed = nn.Embedding(n_classes, img_size * img_size)

        self.net = nn.Sequential(
            # (img_channels+1) × 28 × 28 → 64 × 14 × 14
            nn.Conv2d(img_channels + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 64 × 14 × 14 → 128 × 7 × 7
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            # 128 × 7 × 7 → 256 × 4 × 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
            # 256 × 4 × 4 → 1
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
        )

    def forward(self, imgs, labels):
        label_map = self.label_embed(labels).view(
            -1, 1, self.img_size, self.img_size
        )
        x = torch.cat([imgs, label_map], dim=1)
        return self.net(x).view(-1)


# ---------------------------------------------------------------------------
# WGAN-GP loss
# ---------------------------------------------------------------------------

def gradient_penalty(discriminator, real_imgs, fake_imgs, labels, device):
    """Computes WGAN-GP gradient penalty."""
    B = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    d_interp = discriminator(interpolated, labels)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.view(B, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()
