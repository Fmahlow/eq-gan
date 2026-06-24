"""
WGAN-GP training loop for EQ-GAN.
Supports full mode and all ablation variants.
"""

import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..models.eq_gan import EQGANGenerator, EQGANDiscriminator, gradient_penalty


class EQGANTrainer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        self.generator = EQGANGenerator(
            noise_dim=config["noise_dim"],
            n_classes=config["n_classes"],
            img_channels=config["img_channels"],
            mode=config.get("mode", "full"),
        ).to(device)

        self.discriminator = EQGANDiscriminator(
            img_channels=config["img_channels"],
            n_classes=config["n_classes"],
        ).to(device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.get("lr_g", 1e-4),
            betas=(0.0, 0.9),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.get("lr_d", 1e-4),
            betas=(0.0, 0.9),
        )

        self.lambda_gp   = config.get("lambda_gp", 10.0)
        self.n_critic    = config.get("n_critic", 5)
        self.noise_dim   = config["noise_dim"]
        self.n_classes   = config["n_classes"]

        self.history = {
            "d_loss": [], "g_loss": [], "gp": [],
            "fid": [], "epoch_time": [],
        }

    def _sample_noise(self, n):
        return torch.randn(n, self.noise_dim, device=self.device)

    def _sample_labels(self, n):
        return torch.randint(0, self.n_classes, (n,), device=self.device)

    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()

        d_losses, g_losses, gps = [], [], []

        for batch_idx, (real_imgs, real_labels) in enumerate(dataloader):
            real_imgs   = real_imgs.to(self.device)
            real_labels = real_labels.to(self.device)
            B = real_imgs.size(0)

            # ---- Discriminator step ----
            for _ in range(self.n_critic):
                z = self._sample_noise(B)
                fake_labels = self._sample_labels(B)
                labels_oh   = F.one_hot(fake_labels, self.n_classes).float()

                with torch.no_grad():
                    fake_imgs = self.generator(z, labels_oh)

                d_real = self.discriminator(real_imgs, real_labels)
                d_fake = self.discriminator(fake_imgs, fake_labels)
                gp     = gradient_penalty(
                    self.discriminator, real_imgs, fake_imgs, real_labels, self.device
                )
                d_loss = d_fake.mean() - d_real.mean() + self.lambda_gp * gp

                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()

                d_losses.append(d_loss.item())
                gps.append(gp.item())

            # ---- Generator step ----
            z = self._sample_noise(B)
            fake_labels = self._sample_labels(B)
            labels_oh   = F.one_hot(fake_labels, self.n_classes).float()
            fake_imgs   = self.generator(z, labels_oh)
            g_loss      = -self.discriminator(fake_imgs, fake_labels).mean()

            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
            g_losses.append(g_loss.item())

        return {
            "d_loss": np.mean(d_losses),
            "g_loss": np.mean(g_losses),
            "gp":     np.mean(gps),
        }

    def fit(self, dataloader, n_epochs: int, eval_fn=None, save_dir: str = None,
            eval_every: int = 10):
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(dataloader)
            elapsed = time.time() - t0

            self.history["d_loss"].append(metrics["d_loss"])
            self.history["g_loss"].append(metrics["g_loss"])
            self.history["gp"].append(metrics["gp"])
            self.history["epoch_time"].append(elapsed)

            print(f"Epoch {epoch:03d}/{n_epochs} | "
                  f"D: {metrics['d_loss']:+.4f} | "
                  f"G: {metrics['g_loss']:+.4f} | "
                  f"GP: {metrics['gp']:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if eval_fn is not None and epoch % eval_every == 0:
                fid = eval_fn(self.generator, epoch)
                self.history["fid"].append((epoch, fid))
                print(f"  → FID @ epoch {epoch}: {fid:.2f}")

            if save_dir and epoch % eval_every == 0:
                self.save(save_dir, epoch)

        if save_dir:
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

    def save(self, save_dir: str, epoch: int):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "generator":     self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g":         self.opt_g.state_dict(),
            "opt_d":         self.opt_d.state_dict(),
            "epoch":         epoch,
            "config":        self.config,
            "history":       self.history,
        }, os.path.join(save_dir, f"checkpoint_epoch{epoch:03d}.pt"))

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.history = ckpt.get("history", self.history)
        return ckpt["epoch"]
