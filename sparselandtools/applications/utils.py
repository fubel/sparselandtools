import numpy as np
import imageio
import os

AVAILABLE_IMAGES = ['barbara']


def _add_noise(img, sigma):
    noise = np.random.normal(scale=sigma,
                             size=img.shape).astype(img.dtype)
    return img + noise


def example_image(img_name, noise_std=0):

    imgf = os.path.join('sparselandtools', 'applications', 'assets', img_name + '.png')

    # read image
    try:
        img = imageio.imread(imgf)[:, :, 0].astype('float32')
    except IndexError:
        img = imageio.imread(imgf).astype('float32')

    # add noise
    img = _add_noise(img, sigma=noise_std)
    return img
