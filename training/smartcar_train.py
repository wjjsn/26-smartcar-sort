import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from config.classes import SMARTCAR_CLASSES, IDX_TO_CLASS
from models.cnn import SmartCarCNN
from utils.device import get_device
from utils.transforms import get_smartcar_transform


def get_dataLoaders(batch_size=32, img_size=96):
    """
    创建训练集、验证集和测试集的 DataLoader。

    Args:
        batch_size: 每个批次的样本数量，默认为 32
        img_size: 图像尺寸，默认为 96

    Returns:
        tuple: (train_loader, val_loader, test_loader, IDX_TO_CLASS)
        - train_loader: 训练数据加载器，shuffle=True
        - val_loader: 验证数据加载器，shuffle=False
        - test_loader: 测试数据加载器，shuffle=False
        - IDX_TO_CLASS: 类别索引到类别名的映射字典
    """
    train_transform = get_smartcar_transform(img_size, train=True)
    test_transform = get_smartcar_transform(img_size, train=False)

    train_dataset = datasets.ImageFolder(
        root="data/smartcar/train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root="data/smartcar/val", transform=test_transform
    )
    test_dataset = datasets.ImageFolder(
        root="data/smartcar/test", transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, IDX_TO_CLASS


def train(epochs=50):
    train_loader, val_loader, test_loader, idx_to_class = get_dataLoaders()

    device = get_device()
    model = SmartCarCNN(num_classes=len(SMARTCAR_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"Using device: {device}")
    print(f"Classes: {idx_to_class}")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        accuracy = 100.0 * correct / len(val_loader.dataset)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(
                {"model": model.state_dict(), "idx_to_class": idx_to_class},
                "smartcar_model.pth",
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Val Acc: {accuracy:.2f}% (Best: {best_acc:.2f}%) - Saved best model!",
                end="\r",
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Val Acc: {accuracy:.2f}% (Best: {best_acc:.2f}%)",
                end="\r",
            )


if __name__ == "__main__":
    train()
