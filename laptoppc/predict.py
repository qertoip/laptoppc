import os
from pathlib import Path
from typing import Union

import numpy as np

from helpers import scale_to_0_1, scale_to_0_255

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import matplotlib.cm as colormap
from PIL import Image

from create_model import create_model, model_path

model: Union[keras.Model, None] = None
model_with_heatmap: Union[keras.Model, None] = None


def main():
    load_model()

    LAPTOP_HASHES = [
        '49def37ba21a4448aee1e46ed8885251.jpg',
        '59d5351caf0f420d959690dfdff63f80.jpg',
        'bcc1f8c6f0884717bee38443b8f966fa.jpg',
        '727c54cd2282484caebcb92863b300e5.jpg',
        '804e0752814748cdb7ddf200e049909f.jpg',
        '21a9ee42fff94b6b99185824ec3c70a5.jpg',
        '4ca6812db71043009bbd62d28d61ed34.jpg',
        '0b5deb6cf6ad46a39fedc540cc4168d9.jpg',
        'f04d30c781154a4794b649890939cae1.jpg',
        'f5353761ee044a0d8c4238222b972c2e.jpg',
        'c99a8b4bbe1b44559790a696364bcd3c.jpg',
        'b80f3282e5d8424b940541fde715437c.jpg',
        'c9d86ae93f89464ea471e3af8c4fd8c7.jpg',
        'c8c3a2ab2f5b45d98af2cc34d9418d06.jpg',
    ]
    data_path = Path(__file__).parent.parent / 'data' / 'laptop'
    for image_hash in LAPTOP_HASHES:
        p = predict_for(data_path / image_hash)
        print(p)

    PC_HASHES = [
        '5a8e0e6394f3422dba39ba652ee81fbd.jpg',
        '5ac045988d134cd6bbf4b9c690534710.jpg',
        '5f9bca84ff3d44939f8fdde373a97216.jpg',
        '7eaecc5c6b454255a9540e384571887f.jpg',
        '8b118a624072410380ae62fec9286690.jpg',
        '8d83984513bc4bafba93f9d160c2435b.jpg',
        '08ee1aa72b96483894949430d7b21b9d.jpg',
        '9b94bd7df7ea4bb1957d2e2d15db29c4.jpg',
        '9f42b1b8edb04ac38b5628909f83a97e.jpg',
        '20ebd96e7afe47a19e04222c6b59389a.jpg',
    ]
    data_path = Path(__file__).parent.parent / 'data' / 'pc'
    for image_hash in PC_HASHES:
        p = predict_for(data_path / image_hash)
        print(p)


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


def predict_for(img_path):
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
    img_with_heatmap_overlaid = Image.blend(source_img, heatmap_img_up, 0.75)

    output_path = Path(__file__).parent.parent / 'laptoppc/static/output'
    filename = img_path.stem + '-localized' + img_path.suffix
    filepath = output_path / filename
    img_with_heatmap_overlaid.save(filepath)

    if predicted_class_ix == 0:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    return dict(class_name=class_name, laptop=predictions[0], pc=predictions[1])


def preprocess_for_prediction(img_path):
    # Load image and resize (doesn't keep aspect ratio)
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    # Turn to array of shape (224, 224, 3)
    img = keras.utils.img_to_array(img)
    # Expand array into (1, 224, 224, 3)
    img_batch = np.expand_dims(img, axis=0)
    return img_batch


if __name__ == "__main__":
    main()
