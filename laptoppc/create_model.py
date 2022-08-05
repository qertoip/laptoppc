import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
from keras import layers
from keras import optimizers
from keras.applications.efficientnet_v2 import EfficientNetV2B0


def create_model() -> keras.Model:
    pretrained_base = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=((224, 224, 3))
    )
    pretrained_base.trainable = False

    pooling = keras.layers.GlobalAveragePooling2D()(pretrained_base.output, training=False)
    dropout = keras.layers.Dropout(0.2)(pooling)
    output  = keras.layers.Dense(2, activation='softmax')(dropout)

    new_model = keras.Model(inputs=pretrained_base.input, outputs=output)

    new_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return new_model


def model_path() -> Path:
    p = Path(__file__).parent.parent / 'model'
    p.mkdir(exist_ok=True)
    return p / 'effnet_based_laptop_vs_pc_classifier.h5'
