import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chromatin_tracing_python", # Replace with your own username
    version="0.1",
    author="Kai Sandvold Beckwith",
    author_email="kai.beckwith@embl.de",
    description="Chromatin tracing in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.embl.de/oedegaar/chromatin-team-common-code/-/tree/master/chromatin_tracing_python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
          'h5py',
          'scipy',
          'numpy',
          'pandas',
          'scikit-image',
          'joblib',
          'tifffile',
          'czifile',
          'pyyaml',
          'dask'
      ]
)