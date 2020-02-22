import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="sparselandtools",
    version="1.0.2",
    author="Fabian Herzog",
    author_email="fabian.herzog.dev@gmail.com",
    description="A package for sparse representations and dictionary learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fubel/sparselandtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=requirements,
)
