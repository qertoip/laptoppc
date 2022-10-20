import os
from typing import Union

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import matplotlib.cm as colormap
from PIL import Image

from common import scale_to_0_1, scale_to_0_255, model_path
from create_model import create_model


# The model is global singleton because it's slow to load and warm up
model: Union[keras.Model, None] = None
model_with_heatmap: Union[keras.Model, None] = None


def load_model():
    global model
    model = create_model()
    model.load_weights(model_path())

    # This wrapper is necessary to extract internal layer outputs
    global model_with_heatmap
    model_with_heatmap = keras.Model(
        inputs=model.input,
        outputs=[
            model.layers[-6].output,
            model.layers[-1].output
        ]
    )


def predict_for(img_path) -> dict:
    # The pseudo batch of one image of size (224, 224, 3)
    # Required for model compatibility
    img_batch = preprocess_for_prediction(img_path)
    prediction = model.predict(img_batch)[0]
    if prediction[0] > prediction[1]:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    return dict(class_name=class_name, laptop=prediction[0], pc=prediction[1])


def predict_with_heatmap_for(img_path) -> (dict, Image):
    # The pseudo batch of one image of size (224, 224, 3)
    # Required for model compatibility
    img_batch = preprocess_for_prediction(img_path)

    # Predict
    # Takes 2s the first time, then 50ms
    last_conv_output, predictions = model_with_heatmap.predict(img_batch)

    # Process model output
    last_conv_output = last_conv_output[0]  # (1, 7, 7, 1280) => (7, 7, 1280)
    predictions = predictions[0]
    predicted_class_ix = predictions.argmax()

    # The weights will be necessary to weight convoluted images contributions to the heatmap
    output_layer_weights = model.layers[-1].get_weights()[0]
    output_layer_weights_for_predicted_class = output_layer_weights[:, predicted_class_ix]

    # Heatmap is based on images (feature maps) of the last convolutional layer
    conv_img_h, conv_img_w = last_conv_output.shape[0], last_conv_output.shape[1]   # => (7, 7)
    last_conv_images = last_conv_output.reshape(conv_img_h * conv_img_w, 1280)      # => (49, 1280)

    # Each image contributes proportionally to its (through-GAP) connection weight to the winning class/node in the output layer.
    # The resulting heatmap will be single channel and not normalized.
    #                      matrix @ vector   =>  vector
    #                  (49, 1280) @ (1280,)  =>  (49,)
    heatmap_1d = last_conv_images @ output_layer_weights_for_predicted_class
    heatmap_2d = heatmap_1d.reshape(conv_img_h, conv_img_w)

    # Single channel to color map
    heatmap_color = colormap.viridis(scale_to_0_1(heatmap_2d), bytes=True)[:, :, 0:3]

    # Upscale heatmap using Pillow
    img = img_batch[0]
    heatmap_img = Image.fromarray(scale_to_0_255(heatmap_color))
    heatmap_img_up = heatmap_img.resize(size=img.shape[0:2], resample=Image.Resampling.BILINEAR, reducing_gap=3.0)

    # Overlay heatmap on the source image
    source_img = Image.fromarray(img.astype('uint8'))
    img_with_heatmap_overlaid = Image.blend(source_img, heatmap_img_up, 0.8)

    if predicted_class_ix == 0:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    pred = dict(class_name=class_name, laptop=predictions[0], pc=predictions[1])

    return pred, img_with_heatmap_overlaid


def preprocess_for_prediction(img_path):
    # Load image and resize (doesn't keep aspect ratio)
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    # Turn to array of shape (224, 224, 3)
    img = keras.utils.img_to_array(img)
    # Expand array into (1, 224, 224, 3)
    img_batch = np.expand_dims(img, axis=0)
    return img_batch
