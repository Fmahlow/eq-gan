"""
Evaluation metrics: FID, Inception Score, Precision & Recall.
Uses torchvision InceptionV3 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Always run InceptionV3 on CPU to avoid OOM when multiple
        # experiments run in parallel on the same GPU.
        self.device = torch.device("cpu")
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """
        imgs: (N, C, H, W) in [-1, 1], C can be 1 (grayscale) or 3
        Returns: (N, 2048) numpy array of features
        """
        # Grayscale → RGB
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
        imgs = self.transform(imgs)

        features = []
        ds = TensorDataset(imgs)
        loader = DataLoader(ds, batch_size=batch_size)
        for (batch,) in loader:
            batch = batch.to(self.device)
            feat = self.model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

class InceptionScoreModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device("cpu")
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.eval()
        self.model = model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def get_probs(self, imgs: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = (imgs + 1) / 2
        imgs = self.transform(imgs)

        probs = []
        ds = TensorDataset(imgs)
        for (batch,) in DataLoader(ds, batch_size=batch_size):
            batch = batch.to(self.device)
            logits = self.model(batch)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)


def compute_inception_score(probs: np.ndarray, splits: int = 10):
    """IS = exp(E_x[KL(p(y|x) || p(y))])"""
    N = probs.shape[0]
    split_size = N // splits
    scores = []
    for k in range(splits):
        part = probs[k * split_size: (k + 1) * split_size]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Precision & Recall (Kynkäänniemi et al. 2019)
# ---------------------------------------------------------------------------

def _manifold_estimate(features: np.ndarray, k: int = 3) -> np.ndarray:
    """Estimate manifold as k-NN radii."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(features)
    distances, _ = nbrs.kneighbors(features)
    return distances[:, -1]  # distance to k-th neighbor


def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3,
) -> tuple:
    real_radii = _manifold_estimate(real_features, k)
    fake_radii = _manifold_estimate(fake_features, k)

    from sklearn.neighbors import NearestNeighbors
    real_nbrs = NearestNeighbors(n_neighbors=1).fit(real_features)
    fake_nbrs = NearestNeighbors(n_neighbors=1).fit(fake_features)

    # Precision: fraction of fake samples in real manifold
    dist_fake_to_real, _ = real_nbrs.kneighbors(fake_features)
    precision = (dist_fake_to_real[:, 0] <= real_radii[
        real_nbrs.kneighbors(fake_features)[1][:, 0]
    ]).mean()

    # Recall: fraction of real samples in fake manifold
    dist_real_to_fake, _ = fake_nbrs.kneighbors(real_features)
    recall = (dist_real_to_fake[:, 0] <= fake_radii[
        fake_nbrs.kneighbors(real_features)[1][:, 0]
    ]).mean()

    return float(precision), float(recall)


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, device, n_eval_samples: int = 2000):
        self.device = device
        self.n_eval_samples = n_eval_samples
        self.feature_extractor = InceptionFeatureExtractor(device)
        self.is_model = InceptionScoreModel(device)
        self._real_features_cache = {}

    def cache_real_features(self, dataset_name: str, real_loader):
        """Pre-compute real image features once."""
        real_imgs = []
        for imgs, _ in real_loader:
            real_imgs.append(imgs)
            if sum(x.shape[0] for x in real_imgs) >= self.n_eval_samples:
                break
        real_imgs = torch.cat(real_imgs)[:self.n_eval_samples]
        self._real_features_cache[dataset_name] = \
            self.feature_extractor.extract(real_imgs)
        return self._real_features_cache[dataset_name]

    @torch.no_grad()
    def generate_samples(self, generator, n_classes: int):
        generator.eval()
        samples_per_class = self.n_eval_samples // n_classes
        imgs_list = []
        labels_list = []
        for c in range(n_classes):
            labels = torch.full((samples_per_class,), c, dtype=torch.long,
                                device=self.device)
            imgs = generator.sample(samples_per_class, labels, self.device)
            imgs_list.append(imgs.cpu())
            labels_list.append(labels.cpu())
        generator.train()
        return torch.cat(imgs_list), torch.cat(labels_list)

    def evaluate(self, generator, n_classes: int, dataset_name: str,
                 real_loader=None) -> dict:
        if dataset_name not in self._real_features_cache:
            assert real_loader is not None
            self.cache_real_features(dataset_name, real_loader)
        real_features = self._real_features_cache[dataset_name]

        fake_imgs, _ = self.generate_samples(generator, n_classes)
        fake_features = self.feature_extractor.extract(fake_imgs)

        fid = compute_fid(real_features, fake_features)

        probs = self.is_model.get_probs(fake_imgs)
        is_mean, is_std = compute_inception_score(probs)

        precision, recall = compute_precision_recall(real_features, fake_features)

        return {
            "fid":       fid,
            "is_mean":   is_mean,
            "is_std":    is_std,
            "precision": precision,
            "recall":    recall,
        }
