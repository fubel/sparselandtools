# Sparselandtools

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/1.0.0.dev2/)
[![PyPI](https://img.shields.io/pypi/v/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/sparselandtools.svg?style=flat-square)](https://pypi.org/project/sparselandtools/#description)
[![Build Status](https://travis-ci.com/fubel/sparselandtools.svg?token=e6hQaTqfZFZnG6RmEYXr&branch=master&style=flat-square)](https://travis-ci.com/fubel/sparselandtools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sparselandtools)
![Requires.io](https://img.shields.io/requires/github/fubel/sparselandtools)

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
or use dictionary learning algorithms in small dimensions this ,package is for you.
If you want to use these functions for industrial applications, you should have a
look at more efficient C++-based implementations:

* [The Efficient K-SVD Algorithm by Rubinstein](http://www.cs.technion.ac.il/~ronrubin/software.html)
* [The Efficient K-SVD Denoiser by Lebrun](https://github.com/npd/ksvd)


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


## Contribute

There are a lot of algorithms based on sparse representations and
dictionary learning that are not (yet) included in this package. These
include - among others:

* The Double Sparsity Method
* Trainlets
* Denoiser with Method Noise Post Processing
* Boosted Denoiser with Patch Disagreement

and much more. It would also be interesting to see more applications in this package.
Currently, this package only provides the K-SVD image denoiser [based on the work of
Aharon and Elad](https://www.egr.msu.edu/~aviyente/elad06.pdf). K-SVD can also
be used in many other applications, such as face recognition. Furthermore,
it would be nice to have GPU-versions of all the algorithms available as well.

If you want to see a specific algorithm in this package,
please consider opening a feature request here on Github. If you have written
an algorithm that you think would fit into this package, please fork this
repository, add your algorithm and file a pull request. If something
doesn't work as expected, please open an issue.
