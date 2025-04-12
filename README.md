# Sparselandtools

**Note:** I am no longer maintaining this project so I decided to archive it. 

Sparselandtools is a Python 3 package that provides implementations for
sparse representations and dictionary learning. 

**Note:** I did this project mainly to generate plots for my Master's thesis.
Some of the implementations are more *educational* than *efficient*. If you want
to learn more about sparse representations and dictionary learning using Python,
or use dictionary learning algorithms in small dimensions, this package is for you.
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

## Resources and References

Algorithms are based on pursuit implementations of [scikit-learn](https://scikit-learn.org/stable/) and descriptions in the following papers and projects

1. Aharon, Michal, Michael Elad, and Alfred Bruckstein. "K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation." IEEE Transactions on signal processing 54.11 (2006): 4311-4322.
2. Elad, Michael, and Michal Aharon. "Image denoising via sparse and redundant representations over learned dictionaries." IEEE Transactions on Image processing 15.12 (2006): 3736-3745.
3. Elad, Michael. Sparse and redundant representations: from theory to applications in signal and image processing. Springer Science & Business Media, 2010.
4. [K-SVD Implementation by Rubinstein](http://www.cs.technion.ac.il/~ronrubin/software.html)
5. [The Efficient K-SVD Denoiser by Lebrun](https://github.com/npd/ksvd)

## License

The original code in this repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Note that the implementations are *educational*, and are my own implementations of algorithms presented in published papers and books. This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software. This project relies on various third-party libraries and dependencies, each with their own licensing terms. These dependencies might not be included in the MIT License that covers my original code. Users are responsible for complying with all applicable third-party licenses and are solely responsible for identifying and obtaining any necessary patent licenses for their specific use cases.
