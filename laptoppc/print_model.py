import os
import logging as log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import keras.metrics

from common import model_path
from create_model import create_model


def main():
    setup_logging()
    model: keras.Model = load_model()
    print(model.summary())


def load_model():
    model = create_model()
    model.load_weights(model_path())
    return model


def setup_logging():
    log.basicConfig(level=log.INFO)


if __name__ == '__main__':
    main()
