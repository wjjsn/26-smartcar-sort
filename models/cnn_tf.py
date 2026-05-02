import tensorflow as tf


def create_smartcar_cnn(num_classes=3):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(2, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes),
        ]
    )
    model.build(input_shape=(None, 96, 96, 3))
    return model


class SmartCarCNN(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(SmartCarCNN, self).__init__()
        self.model = create_smartcar_cnn(num_classes)

    def call(self, x, training=False):
        return self.model(x, training=training)
