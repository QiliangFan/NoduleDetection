import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv3D
from tensorflow.keras import Model

def conv333(out, stride=1, activation="relu", data_format="channel_first"):
    return Conv3D(filters=out, kernel_size=(3, 3, 3), activation=activation, strides=stride, padding="same", data_format=data_format)

class BasicBlock(Model):
    def __init__(self, out, stride=1, data_format="channel_first"):
        super(BasicBlock, self).__init__()

        self.conv1 = conv333(out, stride=stride, data_format=data_format)
        self.conv2 = conv333(out, stride=1, activation=None, data_format=data_format)
        self.relu = layers.ReLU()

        if stride == 2:
            self.short_cut = conv333(out, stride=2, data_format=data_format)
        else:
            self.short_cut = None

    def call(self, inputs, training=True):
        if self.short_cut is None:
            out = inputs
        else:
            out = self.short_cut(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += out
        return self.relu(x)

class Resnet3D(Model):
    def __init__(self, layer_size=[3, 4, 6, 3], data_format="channel_first"):
        super(Resnet3D, self).__init__()
        self.layer1 =  keras.Sequential(
            Conv3D(64, kernel_size=(7, 7, 7), strides=2, activation="relu", data_format=data_format, padding="same")
        )

        self.layer2 = keras.Sequential(
            layers.MaxPool3D(pool_size=(3, 3, 3), strides=2, data_format=data_format, padding="same"),
        )
        for i in range(layer_size[0]):
            self.layer2.add(conv333(64))
        
        self.layer3 = keras.Sequential()
        for i in range(layer_size[1]):
            if i == 0:
                self.layer3.add(conv333(128, stride=2))
            else:
                self.layer3.add(conv333(128))

        self.layer4 = keras.Sequential()
        for i in range(layer_size[2]):
            if i == 0 :
                self.layer4.add(conv333(256, stride=2))
            else:
                self.layer4.add(conv333(256))

        self.layer5 = keras.Sequential()
        for i in range(layer_size[3]):
            if i == 0:
                self.layer5.add(conv333(512, stride=2))
            else:
                self.layer5.add(conv333(512, stride=2))

        self.layer6 = keras.Sequential(
            layers.GlobalAveragePooling3D(data_format=data_format),
            layers.Flatten(data_format=data_format),
            Dense(1024, activation="relu"),
            Dense(2, activation="softmax")
        )

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

    def fit(self, data):
        pass
