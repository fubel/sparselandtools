import numpy as np


def dctii(v, normalized=True, sampling_factor=None):
    """
    Computes the inverse discrete cosine transform of type II,
    cf. https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    Args:
        v: Input vector to transform
        normalized: Normalizes the output to make output orthogonal
        sampling_factor: Can be used to "oversample" the input to create overcomplete dictionaries

    Returns:
        Discrete cosine transformed vector
    """
    n = v.shape[0]
    K = sampling_factor if sampling_factor else n
    y = np.array([sum(np.multiply(v, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
    if normalized:
        y[0] = 1 / np.sqrt(2) * y[0]
        y = np.sqrt(2 / n) * y
    return y


def haar(v, sampling_factor=None):
    """
    Compute the Haar transform. The code was modified from
    https://gist.github.com/tristanwietsma/5667982

    Args:
        v: Input vector to transform
        sampling_factor: Can be used to "oversample" the input to create overcomplete dictionaries

    Returns:

    """
    n = v.shape[0]
    K = sampling_factor if sampling_factor else n
    tmp = np.zeros(K)
    count = 2
    while count <= n:
        for i in range(int(count / 2)):
            tmp[2 * i] = (v[i] + v[i + int(count / 2)]) / np.sqrt(2)
            tmp[2 * i + 1] = (v[i] - v[i + int(count / 2)]) / np.sqrt(2)
        for i in range(count):
            v[i] = tmp[i]
        count *= 2
    return np.array(tmp).astype(float)


def dictionary_from_transform(transform, n, K, normalized=True, inverse=True):
    """
    Builds a Dictionary matrix from a given transform

    Args:
        transform: A valid transform (e.g. Haar, DCT-II)
        n: number of rows transform dictionary
        K: number of columns transform dictionary
        normalized: If True, the columns will be l2-normalized
        inverse: Uses the inverse transform (as usually needed in applications)

    Returns:
        Dictionary build from the Kronecker-Delta of the transform applied to the identity.
    """
    H = np.zeros((K, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        H[:, i] = transform(v, sampling_factor=K)
    if inverse:
        H = H.T
    return np.kron(H.T, H.T)


def overcomplete_idctii_dictionary(n, K):
    """
    Build an overcomplete inverse DCT-II dictionary matrix with K > n
    Args:
        n: square of signal dimension
        K: square of desired number of atoms

    Returns:
        Overcomplete DCT-II dictionary
    """
    if K > n:
        return dictionary_from_transform(dctii, n, K, inverse=False)
    else:
        raise ValueError("K needs to be larger than n.")


def unitary_idctii_dictionary(n):
    """
    Build a unitary inverse DCT-II dictionary matrix with K = n
    Args:
        n: square of signal dimension

    Returns:
        Unitary DCT-II dictionary
    """
    return dictionary_from_transform(dctii, n, n, inverse=False)


def overcomplete_haar_dictionary(n, K):
    """
    Build an overcomplete inverse Haar dictionary matrix with K > n
    Args:
        n: square of signal dimension
        K: square of desired number of atoms

    Returns:
        Overcomplete Haar dictionary
    """
    if K > n:
        return dictionary_from_transform(haar, n, K)
    else:
        raise ValueError("K needs to be larger than n.")


def unitary_haar_dictionary(n):
    """
    Build a unitary inverse Haar dictionary matrix with K = n
    Args:
        n: square of signal dimension

    Returns:
        Unitary Haar dictionary
    """
    return dictionary_from_transform(haar, n, n)


def random_dictionary(n, K, normalized=True, seed=None):
    """
    Build a random dictionary matrix with K = n
    Args:
        n: square of signal dimension
        K: square of desired dictionary atoms
        normalized: If true, columns will be l2-normalized
        seed: Random seed

    Returns:
        Random dictionary
    """
    if seed:
        np.random.seed(seed)
    H = np.random.rand(n, K) * 255
    if normalized:
        for k in range(K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])
    return np.kron(H, H)
