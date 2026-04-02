import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import cv2

from config.classes import SMARTCAR_CLASSES, IDX_TO_CLASS
from models.cnn import SmartCarCNN
from utils.device import get_device
from utils.transforms import get_smartcar_predict_transform


def predict_image(model, img_path, idx_to_class, device):
    transform = get_smartcar_predict_transform(96)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load {img_path}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()

    prob_values = probs.squeeze().cpu().numpy()
    print(f"置信度:")
    for i, cls_name in idx_to_class.items():
        print(f"  {cls_name}: {prob_values[i]:.4f}")

    return idx_to_class[pred]


def main():
    checkpoint = torch.load("smartcar_model.pth", weights_only=False)
    idx_to_class = checkpoint["idx_to_class"]

    device = get_device()
    model = SmartCarCNN(num_classes=len(SMARTCAR_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model"])

    print(f"Using device: {device}")
    print(f"Classes: {idx_to_class}")

    test_dir = Path("data/smartcar/test")
    categories = SMARTCAR_CLASSES

    correct = 0
    total = 0

    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        for img_path in cat_dir.glob("*.png"):
            pred = predict_image(model, img_path, idx_to_class, device)
            true_label = cat
            is_correct = pred == true_label
            correct += is_correct
            total += 1
            status = "✓" if is_correct else "✗"
            print(f"{status} {img_path.name}: predicted={pred}, actual={true_label}")

    print(f"\nAccuracy: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
