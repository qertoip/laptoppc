import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF logging (must be run before importing tf or keras)
import tensorflow
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0

from tools import model_path, root_path, setup_logging
from create_model import create_model


def main():
    setup_logging()
    pretrained_base = EfficientNetV2B0(
        weights='imagenet',
        include_top=True,
        input_shape=((224, 224, 3))
    )
    path = root_path() / 'model' / 'base_model.png'
    visualize_model(pretrained_base, path)

    finetuned = create_model()
    finetuned.load_weights(model_path())
    path = root_path() / 'model' / 'finetuned_model.png'
    visualize_model(finetuned, path)


def visualize_model(pretrained_base, path):
    tensorflow.keras.utils.plot_model(
        pretrained_base,
        to_file=path,
        show_shapes=True,
        show_dtype=True,
        show_layer_activations=True,
        show_layer_names=True,
        dpi=360
        # rankdir='LR'  # left-to-right
    )
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
