import os
import imghdr
from pathlib import Path
from typing import Union

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras

from create_model import create_model, model_path

UPLOAD_FOLDER = Path(__file__).parent / 'static/uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

model: Union[keras.Model, None] = None
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
        prediction = predict_for(path)
        return render_template('upload.html', filename=filename, prediction=prediction)
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


# This is used in development when this file is run directly
if __name__ == "__main__":
    development_main()
