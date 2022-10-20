import os
import imghdr
from pathlib import Path
from typing import Union

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from predict import load_model, predict_with_heatmap_for

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import keras
from PIL import Image

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
        dir = Path(flask.config['UPLOAD_FOLDER'])
        dir.mkdir(exist_ok=True, parents=True)
        path = dir / filename
        file.save(path)

        filetype = imghdr.what(path)
        if filetype != 'jpeg':
            flash(f'Not a real JPEG, please try another file.')
            path.unlink()
            return redirect(request.url)

        #flash('Success!!')
        prediction, heatmap = predict_with_heatmap_for(path)
        heatmap_filename = save_aside(heatmap, path)
        return render_template(
            'upload.html',
            filename=filename,
            localized_filename=heatmap_filename,
            prediction=prediction
        )
    else:
        flash(f'Only JPEG-s are allowed for simplicity and security. Sorry!')
        return redirect(request.url)


@flask.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def save_aside(img: Image, path: Path) -> str:
    filename = path.stem + '-localized' + path.suffix
    filepath = path.parent / filename
    img.save(filepath)
    return str(filename)


# This is used in development when this file is run directly
if __name__ == "__main__":
    development_main()
