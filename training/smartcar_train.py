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
    创建训练集和测试集的 DataLoader。

    Args:
        batch_size: 每个批次的样本数量，默认为 32
        img_size: 图像尺寸，默认为 96

    Returns:
        tuple: (train_loader, test_loader, IDX_TO_CLASS)
        - train_loader: 训练数据加载器，shuffle=True
        - test_loader: 测试数据加载器，shuffle=False
        - IDX_TO_CLASS: 类别索引到类别名的映射字典
    """
    # 分别为训练集和测试集创建不同的预处理流水线
    # 训练集使用带随机增强的 transform（旋转、颜色抖动、水平翻转）
    # 测试集使用确定性的 transform（无随机性，保证结果可重复）
    train_transform = get_smartcar_transform(img_size, train=True)
    test_transform = get_smartcar_transform(img_size, train=False)

    train_dataset = datasets.ImageFolder(
        root="data/smartcar/train", transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root="data/smartcar/test", transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, IDX_TO_CLASS


def train(epochs=20):
    train_loader, test_loader, idx_to_class = get_dataLoaders()

    device = get_device()
    model = SmartCarCNN(num_classes=len(SMARTCAR_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Using device: {device}")
    print(f"Classes: {idx_to_class}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

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

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Acc: {accuracy:.2f}%",
            end='\r'
        )

    torch.save(
        {"model": model.state_dict(), "idx_to_class": idx_to_class},
        "smartcar_model.pth",
    )
    print("Model saved to smartcar_model.pth")


if __name__ == "__main__":
    train()
