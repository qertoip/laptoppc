import os
from pathlib import Path
import logging as log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import keras.metrics

from create_model import create_model


def main():
    setup_logging()
    model: keras.Model = load_model()
    print(model.summary())


def load_model():
    model = create_model()
    model.load_weights(model_path())
    return model


def model_path():
    p = Path(__file__).parent.parent / 'model'
    p.mkdir(exist_ok=True)
    return p / 'effnet_based_laptop_vs_pc_classifier.h5'


def setup_logging():
    log.basicConfig(level=log.INFO)


if __name__ == '__main__':
    main()
