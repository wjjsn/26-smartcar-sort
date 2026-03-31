import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_smartcar_dataLoaders(batch_size=16, img_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    CLASSES = ["交通工具-直行", "武器-左", "物资-右"]
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

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
