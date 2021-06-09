# Sparselandtools

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/1.0.0.dev2/)
[![PyPI](https://img.shields.io/pypi/v/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/#description)
[![Build Status](https://travis-ci.com/fubel/sparselandtools.svg?token=e6hQaTqfZFZnG6RmEYXr&branch=master&style=flat-square)](https://travis-ci.com/fubel/sparselandtools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sparselandtools)

Sparselandtools is a Python 3 package that provides implementations for
sparse representations and dictionary learning. In particular, it
includes implementations for

**For Sparse Representations:**
* Matching Pursuit
* Orthogonal Matching Pursuit
* Thresholding Pursuit
* Basis Pursuit

**For Dictionaries in General:**
* Mutual Coherence
* DCT Dictionary
* Haar Dictionary
* Overcomplete DCT Dictionary
* Visualization Tools for Dictionaries

**For Dictionary Learning:**
* K-SVD Algorithm
* Approximate K-SVD Algorithm

**For Application:**
* Approximate K-SVD Image Denoiser

**Note:** I did this project mainly to generate plots for my Master's thesis.
Some of the implementations are more *educational* than *efficient*. If you want
to learn more about sparse representations and dictionary learning using Python,
or use dictionary learning algorithms in small dimensions, this package is for you.
If you want to use these functions for industrial applications, you should have a
look at more efficient C++-based implementations:

* [The Efficient K-SVD Algorithm by Rubinstein](http://www.cs.technion.ac.il/~ronrubin/software.html)
* [The Efficient K-SVD Denoiser by Lebrun](https://github.com/npd/ksvd)

**Warning:** I'm currently pursuing my PhD in computer vision and have no time to keep this project up-to-date. Any pull requests for new features or bug fixes are welcome! 


## Getting Started

Sparselandtools is available as a PyPI package. You can install it using

```
pip install sparselandtools
```

![DCT and Haar Dictionary](https://snag.gy/h7Il2j.jpg)

The following code creates a redundant (=overcomplete) DCT-II dictionary
and plots it. It also prints out the dictionaries mutual coherence.

```python
from sparselandtools.dictionaries import DCTDictionary
import matplotlib.pyplot as plt

# create dictionary
dct_dictionary = DCTDictionary(8, 11)

# plot dictionary
plt.imshow(dct_dictionary.to_img())
plt.show()

# print mutual coherence
print(dct_dictionary.mutual_coherence())
```

More examples can be found in the corresponding Jupyter Notebook.


## Reference 

If you find my implementations useful for your academic projects, feel free to cite

```
@software{fabian_herzog_2021_4916395,
  author       = {Fabian Herzog},
  title        = {{sparselandtools: A Python package for sparse 
                   representations and dictionary learning, including
                   matching pursuit, K-SVD and applications.}},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4916395},
  url          = {https://doi.org/10.5281/zenodo.4916395}
}
```
