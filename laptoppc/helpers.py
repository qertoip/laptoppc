import numpy as np
import matplotlib.pyplot as plt

def scale_to_0_1(array: np.array) -> np.array:
    return (array - array.min()) / (array.max() - array.min())


def scale_to_0_255(array: np.array) -> np.array:
    return (scale_to_0_1(array) * 255).astype('uint8')


# Debug show image using matplot machinery
def ishow(image):
    plt.imshow(image, cmap='gray')
    plt.show()
