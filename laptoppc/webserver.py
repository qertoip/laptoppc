import os
import imghdr
from pathlib import Path
from typing import Union

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np

from helpers import scale_to_0_1, scale_to_0_255

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
import matplotlib.cm as colormap
from PIL import Image

from create_model import create_model, model_path

UPLOAD_FOLDER = Path(__file__).parent / 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

model: Union[keras.Model, None] = None
model_with_heatmap: Union[keras.Model, None] = None

flask: Flask = Flask(__name__)


# Entry point for gunicorn in production (must return WSGI app)
def production_main() -> Flask:
    before_wsgi_app()
    return flask  # must return WSGI app to satisfy gunicorn


# Entry point for development (no gunicorn, bare Flask)
def development_main():
    before_wsgi_app()
    flask.run()


def before_wsgi_app():
    flask.secret_key = "DpNTqr9g9yCQDDdyHbrAoi2D"
    flask.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    flask.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    load_model()


def allowed_file_ext(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@flask.route('/')
def upload_form():
    return render_template('upload.html')


@flask.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file_ext(file.filename):
        filename = secure_filename(file.filename)
        path = Path(flask.config['UPLOAD_FOLDER']) / filename
        file.save(path)

        filetype = imghdr.what(path)
        if filetype != 'jpeg':
            flash(f'Not a real JPEG, please try another file.')
            path.unlink()
            return redirect(request.url)

        #flash('Success!!')
        prediction, localized_filename = predict_with_heatmap_for(path)
        return render_template(
            'upload.html',
            filename=filename,
            localized_filename=localized_filename,
            prediction=prediction
        )
    else:
        flash(f'Only JPEG-s are allowed for simplicity and security. Sorry!')
        return redirect(request.url)


@flask.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


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


def preprocess_for_prediction(path):
    # Load image and resize (doesn't keep aspect ratio)
    img = keras.utils.load_img(path, target_size=(224, 224))
    # Turn to array of shape (224, 224, 3)
    img = keras.utils.img_to_array(img)
    # Expand array into (1, 224, 224, 3)
    img_batch = np.expand_dims(img, 0)
    return img_batch


def predict_for(path):
    img_batch = preprocess_for_prediction(path)
    p = model.predict(img_batch)[0]
    if p[0] > p[1]:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    return dict(class_name=class_name, laptop=p[0], pc=p[1])


def predict_with_heatmap_for(img_path):
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

    output_path = Path(__file__).parent.parent / 'laptoppc/static/uploads'
    filename = img_path.stem + '-localized' + img_path.suffix
    filepath = output_path / filename
    img_with_heatmap_overlaid.save(filepath)

    if predicted_class_ix == 0:
        class_name = 'Laptop'
    else:
        class_name = 'Desktop PC'
    return dict(class_name=class_name, laptop=predictions[0], pc=predictions[1]), filename


# This is used in development when this file is run directly
if __name__ == "__main__":
    development_main()
