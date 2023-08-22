import os
import logging as log
from datetime import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import tensorflow as tf
import keras
import keras.metrics
import keras.utils
import keras.optimizers
import keras.callbacks


from common import data_path, model_path, root_path
from create_model import create_model


def main():
    setup_logging()
    train_set, test_set = read_dataset()
    model = create_model()
    train_model(model, train_set, test_set)  # autosave


def read_dataset():
    train_set: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        data_path(),
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
        subset='training',
        seed=13047  # must be same for training and test sets
    )

    test_set: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        data_path(),
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
        subset='validation',
        seed=13047  # must be same for training and test sets
    )

    return train_set, test_set


def train_model(model: keras.Model, train_set, test_set):
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path(), save_best_only=True)

    log_dir: Path = root_path() / 'log' / 'fit' / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_set,
        epochs=8,
        callbacks=[checkpoint, tensorboard_callback],
        validation_data=test_set
    )
    model.save(filepath=model_path())


def setup_logging():
    log.basicConfig(level=log.INFO)


if __name__ == '__main__':
    main()
