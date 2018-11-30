import warnings
import numpy as np

from sparselandtools.dictionaries.utils import overcomplete_haar_dictionary, overcomplete_idctii_dictionary, \
    random_dictionary, unitary_haar_dictionary, unitary_idctii_dictionary


class Dictionary:
    """
    The Dictionary class is more or less a wrapper around the numpy array class. It holds a numpy ndarray in
    the attribute `matrix` and adds some useful functions for it. The dictionary elements can be accessed
    either by D.matrix[i,j] or directly through D[i,j].

    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.shape = matrix.shape

    def __getitem__(self, item):
        return self.matrix[item]

    def is_unitary(self):
        """
        Checks whether the dictionary is unitary.

        Returns:
            True, if the dicitonary is unitary.
        """
        n, K = self.shape
        if n == K:
            return np.allclose(np.dot(self.matrix.T, self.matrix), np.eye(n))
        else:
            return False

    def is_normalized(self):
        """
        Checks wheter the dictionary is l2-normalized.

        Returns:
            True, if dictionary is l2-normalized.
        """
        n, K = self.shape
        return np.allclose([np.linalg.norm(self.matrix[:, i]) for i in range(K)], np.ones(K))

    def as_patches(self):
        """
        Depcretaed.
        """
        warnings.warn("as_patches() will be removed in the next version, use to_img() instead.", FutureWarning)
        return self.to_img()

    def mutual_coherence(self):
        """
        Computes the dictionary's mutual coherence.

        Returns:
            Mutual coherence
        """
        return np.max(self._mutual_coherence(self.matrix))

    @staticmethod
    def _mutual_coherence(D):
        n, K = D.shape
        mu = [np.abs(np.dot(D[:, i].T, D[:, j]) /
                     (np.linalg.norm(D[:, i]) * np.linalg.norm(D[:, j])))
              for i in range(K) for j in range(K) if j != i]
        return mu

    def to_img(self):
        """
        Transforms the dictionary columns into patches and orders them for plotting purposes.

        Returns:
            Reordered dictionary matrix
        """
        # dictionary dimensions
        D = self.matrix
        n, K = D.shape
        M = self.matrix
        # stretch atoms
        for k in range(K):
            M[:, k] = M[:, k] - (M[:, k].min())
            if M[:, k].max():
                M[:, k] = M[:, k] / D[:, k].max()

        # patch size
        n_r = int(np.sqrt(n))

        # patches per row / column
        K_r = int(np.sqrt(K))

        # we need n_r*K_r+K_r+1 pixels in each direction
        dim = n_r * K_r + K_r + 1
        V = np.ones((dim, dim)) * np.min(D)

        # compute the patches
        patches = [np.reshape(D[:, i], (n_r, n_r)) for i in range(K)]

        # place patches
        for i in range(K_r):
            for j in range(K_r):
                V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
                    i * K_r + j]
        return V


class DCTDictionary(Dictionary):
    """
    A Dictionary based on the IDCT-II transform
    """

    def __init__(self, n, K):
        if n == K:
            D = unitary_idctii_dictionary(n)
        elif n < K:
            D = overcomplete_idctii_dictionary(n, K)
        else:
            raise ValueError("K has to be as least as large as n.")
        super().__init__(D)


class HaarDictionary(Dictionary):
    """
    A Dictionary based on the inverse Haar transform
    """

    def __init__(self, n, K):
        if n == K:
            D = unitary_haar_dictionary(n)
        elif n < K:
            D = overcomplete_haar_dictionary(n, K)
        else:
            raise ValueError("K has to be as least as large as n.")
        super().__init__(D)


class RandomDictionary(Dictionary):
    """
    A random Dictionary
    """

    def __init__(self, n, K):
        if n == K:
            D = random_dictionary(n, n)
        elif n < K:
            D = random_dictionary(n, K)
        else:
            raise ValueError("K has to be as least as large as n.")
        super().__init__(D)
