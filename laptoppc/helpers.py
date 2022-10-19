import numpy as np


def scale_to_0_1(array: np.array) -> np.array:
    return (array - array.min()) / (array.max() - array.min())
