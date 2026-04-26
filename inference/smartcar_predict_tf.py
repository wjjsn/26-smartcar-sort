import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import tensorflow as tf
from PIL import Image

from config.classes import SMARTCAR_CLASSES


def predict_image(model, img_path, idx_to_class):
    img = Image.open(str(img_path)).convert("RGB")
    img = img.resize((96, 96))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    output = model.predict(img_array, verbose=0)[0]
    probs = tf.nn.softmax(output).numpy()
    pred = int(np.argmax(output))

    print(f"置信度:")
    for i, cls_name in idx_to_class.items():
        print(f"  {cls_name}: {probs[int(i)]:.4f}")

    return idx_to_class[str(pred)]


def main():
    from models.cnn_tf import create_smartcar_cnn

    model = create_smartcar_cnn(num_classes=len(SMARTCAR_CLASSES))
    model.load_weights("smartcar_model_tf.weights.h5")

    with open("idx_to_class_tf.json", "r") as f:
        idx_to_class = json.load(f)

    print(f"Classes: {idx_to_class}")

    test_dir = Path("data/smartcar/test")
    categories = SMARTCAR_CLASSES

    correct = 0
    total = 0

    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        for img_path in list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")):
            pred = predict_image(model, img_path, idx_to_class)
            true_label = cat
            is_correct = pred == true_label
            correct += is_correct
            total += 1
            status = "✅" if is_correct else "❌"
            print(f"{status} {img_path.name}: predicted={pred}, actual={true_label}")

            print("-" * 40)

    print(f"\nAccuracy: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
