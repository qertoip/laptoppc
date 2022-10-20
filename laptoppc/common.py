from pathlib import Path

import numpy as np


def root_path():
    return Path(__file__).parent.parent


def data_path():
    path = root_path() / 'data'
    path.mkdir(exist_ok=True)
    return path


def model_path():
    path = root_path() / 'model'
    path.mkdir(exist_ok=True)
    return path / 'effnet_based_laptop_vs_pc_classifier.h5'


def scale_to_0_1(array: np.array) -> np.array:
    return (array - array.min()) / (array.max() - array.min())


def scale_to_0_255(array: np.array) -> np.array:
    return (scale_to_0_1(array) * 255).astype('uint8')


# Debug show image using matplot machinery
def ishow(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.show()
