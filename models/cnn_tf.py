import tensorflow as tf


class SmartCarCNN(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(SmartCarCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=3, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=3, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            128, kernel_size=3, padding="same", activation="relu"
        )
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(self.fc1(x), training=training)
        x = self.fc2(x)
        return x
