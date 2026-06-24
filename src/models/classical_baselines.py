"""
Classical GAN baselines: DCGAN and WGAN-GP (conditional).
These run fully on GPU — much faster than quantum variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim: int = 64, n_classes: int = 10,
                 img_channels: int = 1):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.net = nn.Sequential(
            # (noise_dim + n_classes) → 512×4×4
            nn.Linear(noise_dim + n_classes, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            # 4×4 → 7×7
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

    def forward(self, z, labels):
        label_enc = self.label_emb(labels)
        x = torch.cat([z, label_enc], dim=1)
        return self.net(x)

    def sample(self, n: int, labels: torch.Tensor, device: torch.device):
        z = torch.randn(n, self.noise_dim, device=device)
        return self(z, labels)


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels: int = 1, n_classes: int = 10,
                 img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.label_embed = nn.Embedding(n_classes, img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
        )

    def forward(self, imgs, labels):
        label_map = self.label_embed(labels).view(
            -1, 1, self.img_size, self.img_size)
        x = torch.cat([imgs, label_map], dim=1)
        return self.net(x).view(-1)


def gradient_penalty_classical(disc, real, fake, labels, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d = disc(interp, labels)
    grad = torch.autograd.grad(d, interp,
                               grad_outputs=torch.ones_like(d),
                               create_graph=True, retain_graph=True)[0]
    return ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
