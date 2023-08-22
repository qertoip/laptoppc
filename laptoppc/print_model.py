import os
import logging as log

import pydot
import graphviz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import keras.metrics
import keras.utils
from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.layers.convolutional.conv2d import Conv2D

from common import model_path, root_path
from create_model import create_model


def main():
    setup_logging()

    pretrained_base = EfficientNetV2B0(
        weights='imagenet',
        include_top=True,
        input_shape=((224, 224, 3))
    )
    l: Conv2D = pretrained_base.layers[12]
    print(l.name)
    print(l.kernel_size)
    # for layer in pretrained_base.layers:
    #     print(layer.)

    # model: keras.Model = load_model()
    # print(model.summary())
    # keras.utils.plot_model(
    #     model,
    #     to_file=root_path() / 'model' / 'model.png',
    #     show_shapes=True,
    #     show_dtype=True,
    #     show_layer_activations=True,
    #     show_layer_names=True,
    #     dpi=360
    #     #rankdir='LR'  # left-to-right
    # )


def load_model():
    model = create_model()
    model.load_weights(model_path())
    return model


def setup_logging():
    log.basicConfig(level=log.INFO)


if __name__ == '__main__':
    main()
