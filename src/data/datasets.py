"""Dataset loaders for MNIST, FashionMNIST, and BreastMNIST."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


IMG_SIZE = 28


def _base_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
    ])


def get_mnist(batch_size: int = 64, data_root: str = "./data"):
    transform = _base_transform()
    train = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    n_classes = 10
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        n_classes,
    )


def get_fashion_mnist(batch_size: int = 64, data_root: str = "./data"):
    transform = _base_transform()
    train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    n_classes = 10
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        n_classes,
    )


def get_breast_mnist(batch_size: int = 64, data_root: str = "./data"):
    """
    BreastMNIST: 2-class ultrasound dataset (benign vs. malignant).
    Part of MedMNIST v2.
    """
    info = INFO["breastmnist"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train = DataClass(split="train", transform=transform, download=True, root=data_root)
    test  = DataClass(split="test",  transform=transform, download=True, root=data_root)

    # MedMNIST labels are (N,1) arrays; wrap to return flat tensors
    class FlatLabelDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return img, torch.tensor(label).squeeze().long()

    return (
        DataLoader(FlatLabelDataset(train), batch_size=batch_size, shuffle=True,
                   num_workers=2, pin_memory=True),
        DataLoader(FlatLabelDataset(test),  batch_size=batch_size, shuffle=False,
                   num_workers=2, pin_memory=True),
        n_classes,
    )


DATASET_REGISTRY = {
    "mnist":         get_mnist,
    "fashionmnist":  get_fashion_mnist,
    "breastmnist":   get_breast_mnist,
}


def get_dataset(name: str, batch_size: int = 64, data_root: str = "./data"):
    assert name in DATASET_REGISTRY, f"Unknown dataset: {name}"
    return DATASET_REGISTRY[name](batch_size=batch_size, data_root=data_root)
