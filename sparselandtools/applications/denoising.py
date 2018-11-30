import logging
from typing import Type

import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sparselandtools.dictionaries import Dictionary
from sparselandtools.learning import ApproximateKSVD
from sparselandtools.pursuits import Pursuit

logging.basicConfig(level=logging.INFO)


class KSVDImageDenoiser(ApproximateKSVD):
    """
    A denoiser based on the approximate K-SVD.
    """

    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit]):

        super().__init__(dictionary, pursuit, 0)
        self.multiplier = None
        self.n_iter = None
        self.image = None
        self.patch_size = None
        self.image_size = None
        self.image_root = None

    def denoise(self, image, sigma=3, multiplier=10, n_iter=15, patch_size=8, noise_gain=1.15):
        # promote values to super
        self.noise_gain = noise_gain
        self.sigma = sigma

        # error handling
        if image.shape[0] != image.shape[1]:
            raise ValueError("Image must be square!")

        # set initial values
        self.image = image
        self.sigma = sigma
        self.multiplier = multiplier
        self.n_iter = n_iter
        self.patch_size = patch_size

        # compute further values
        self.image_size = image.shape[0]

        # prepare K-SVD
        patches = extract_patches_2d(self.image, (self.patch_size, self.patch_size))
        Y = np.array([p.reshape(self.patch_size**2) for p in patches]).T

        # iterate K-SVD
        for itr in range(self.n_iter):
            self.sparse_coding(Y)
            self.dictionary_update(Y)

        # reconstruct image
        # this was translated from the Matlab code in Michael Elads book
        # cf. Elad, M. (2010). Sparse and redundant representations:
        # from theory to applications in signal and image processing. New York: Springer.
        out = np.zeros(image.shape)
        weight = np.zeros(image.shape)
        logging.info("reconstructing")
        i = j = 0
        for k in range((self.image_size - self.patch_size + 1) ** 2):
            patch = np.reshape(np.matmul(self.dictionary.matrix, self.alphas[:, k]), (self.patch_size, self.patch_size))
            out[j:j + self.patch_size, i:i + self.patch_size] += patch
            weight[j:j + self.patch_size, i:i + self.patch_size] += 1
            if i < self.image_size - self.patch_size:
                i += 1
            else:
                i = 0
                j += 1
        out = np.divide(out + self.multiplier * self.image, weight + self.multiplier)
        return out, self.dictionary, self.alphas
