import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="looptrace",
    version="0.2",
    author="Kai Sandvold Beckwith",
    author_email="kai.beckwith@embl.de",
    description="Fitting and analysis of chromatin tracing data in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.embl.de/grp-ellenberg/looptrace",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[]
)
