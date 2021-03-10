import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)

layer = tf.keras.layers.Dense(10)
input = tf.ones([10, 20])
out = layer(input)