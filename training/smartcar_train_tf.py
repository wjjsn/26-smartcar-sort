import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from config.classes import SMARTCAR_CLASSES
from models.cnn_tf import SmartCarCNN


def get_datasets(batch_size=32, img_size=96):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/smartcar/train",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "data/smartcar/test",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    return train_ds, test_ds, idx_to_class


def train(epochs=20):
    train_ds, test_ds, idx_to_class = get_datasets()

    model = SmartCarCNN(num_classes=len(SMARTCAR_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print(f"Classes: {idx_to_class}")
    print(f"Train batches: {len(train_ds)}, Test batches: {len(test_ds)}")

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1,
    )

    model.save("smartcar_model_tf.h5")
    model.save_weights("smartcar_model_tf.weights.h5")
    print("Model saved to smartcar_model_tf.h5")
    print("Weights saved to smartcar_model_tf.weights.h5")

    import json

    with open("idx_to_class_tf.json", "w") as f:
        json.dump(idx_to_class, f)
    print("Index to class mapping saved to idx_to_class_tf.json")


if __name__ == "__main__":
    train()
