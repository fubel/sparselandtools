import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparselandtools",
    version="1.0.1",
    author="Fabian Herzog",
    author_email="fabian.herzog.dev@gmail.com",
    description="A package for sparse representations and dictionary learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fubel/py-sparselandtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        "brewer2mpl>=1.4.1",
        "cloudpickle>=0.5.3",
        "cycler>=0.10.0",
        "Cython>=0.29",
        "dask>=0.18.2",
        "decorator>=4.3.0",
        "imageio>=2.3.0",
        "kiwisolver>=1.0.1",
        "llvmlite>=0.25.0",
        "matplotlib>=2.2.2",
        "networkx>=2.1",
        "numpy>=1.15.0",
        "pandas>=0.23.4",
        "Pillow>=5.2.0",
        "prettyplotlib>=0.1.7",
        "pyparsing>=2.2.0",
        "pytest>=3.10.0",
        "python-dateutil>=2.7.3",
        "pytz>=2018.5",
        "scikit-image>=0.14.0",
        "scikit-learn>=0.19.2",
        "scipy>=1.1.0",
        "seaborn>=0.9.0",
        "six>=1.11.0",
        "toolz>=0.9.0",
        "tqdm>=4.28.1",
    ]
)
