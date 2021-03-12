import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
for i in range(10):
    with tf.device(f"/gpu:{i%2}"):
        layer = tf.keras.layers.Dense(10)
        input = tf.ones([10, 20])
        out = layer(input)

