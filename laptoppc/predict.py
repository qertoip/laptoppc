import os
from pathlib import Path
from typing import Union

import numpy as np

from helpers import scale_to_0_1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import scipy
from PIL import Image

from create_model import create_model, model_path

model: Union[keras.Model, None] = None
model2: Union[keras.Model, None] = None


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
    global model2
    model2 = keras.Model(
        inputs=model.input,
        outputs=[
            model.layers[-6].output,
            model.layers[-1].output
        ]
    )


def predict_for(path):
    img_batch = preprocess_for_prediction(path)

    # Predict
    last_conv_output, pred_vec = model2.predict(img_batch)

    # Process output
    last_conv_output = np.squeeze(last_conv_output)  # (1, 7, 7, 1280) => (7, 7, 1280)
    pred_vec = pred_vec.flatten()
    predicted_class_ix = np.argmax(pred_vec)

    last_layer_weights = model.layers[-1].get_weights()[0]
    last_layer_weights_for_predicted_class = last_layer_weights[:, predicted_class_ix]

    last_conv_img_h = last_conv_output.shape[0]
    last_conv_img_w = last_conv_output.shape[1]

    # Heatmap is based on images (feature maps) of the last convolutional layer.
    # Each image contributes proportionally to its connection weight to predicted class/node in the output layer.
    # The resulting heatmap will be single channel and not be normalized.
    heatmap_1d = np.dot(
        last_conv_output.reshape(last_conv_img_h * last_conv_img_w, 1280),
        last_layer_weights_for_predicted_class
    )
    heatmap_2d = heatmap_1d.reshape(last_conv_img_h, last_conv_img_w)

    # Upscale
    img = img_batch[0]
    h = int(img.shape[0] / last_conv_img_h)
    w = int(img.shape[1] / last_conv_img_w)
    heatmap_2d_up = scipy.ndimage.zoom(heatmap_2d, (h, w), order=1)

    # Single channel to color map
    heatmap_2d_color = colormap.viridis(scale_to_0_1(heatmap_2d_up), bytes=True)[:, :, 0:3]

    output_path = Path(__file__).parent.parent / 'laptoppc/static/output'

    pill_heatmap = Image.fromarray(heatmap_2d_color)
    #filename = path.stem + '-heatmap' + path.suffix
    #filepath = output_path / filename
    #pill_heatmap.save(filepath)

    pill_img = Image.fromarray(img.astype('uint8'))
    pill_blend = Image.blend(pill_img, pill_heatmap, 0.75)
    filename = path.stem + '-localized' + path.suffix
    filepath = output_path / filename
    pill_blend.save(filepath)

    if predicted_class_ix == 0:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    return dict(class_name=class_name, laptop=pred_vec[0], pc=pred_vec[1])


def preprocess_for_prediction(path):
    # Load image and resize (doesn't keep aspect ratio)
    img = keras.utils.load_img(path, target_size=(224, 224))
    # Turn to array of shape (224, 224, 3)
    img = keras.utils.img_to_array(img)
    # Expand array into (1, 224, 224, 3)
    img_batch = np.expand_dims(img, axis=0)
    return img_batch


if __name__ == "__main__":
    main()
