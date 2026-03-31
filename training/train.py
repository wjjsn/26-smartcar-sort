import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from models.cnn import MNISTCNN
from utils.device import get_device
from utils.transforms import MNIST_TRAIN_TRANSFORM


def train():
    transform = MNIST_TRAIN_TRANSFORM

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = get_device()
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Using device: {device}")

    for epoch in range(5):
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
            f"Epoch {epoch + 1}/5 - Loss: {total_loss / len(train_loader):.4f} - Acc: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")


if __name__ == "__main__":
    train()
