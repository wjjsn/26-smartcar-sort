import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from config.classes import SMARTCAR_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX
from utils.transforms import get_smartcar_transform


def get_smartcar_dataLoaders(batch_size=16, img_size=32):
    transform = get_smartcar_transform(img_size)

    train_dataset = datasets.ImageFolder(
        root="data/smartcar/train", transform=transform
    )
    test_dataset = datasets.ImageFolder(root="data/smartcar/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, IDX_TO_CLASS


if __name__ == "__main__":
    train_loader, test_loader, idx_to_class = get_smartcar_dataLoaders()
    print(f"Classes: {idx_to_class}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    for imgs, labels in train_loader:
        print(f"Batch shape: {imgs.shape}, Labels: {labels}")
        break
