# Chromatin tracing in python

Python is a Python library for performing steps of chromatin tracing in Python.

## Installation

The simplest way to install the package and all dependencies is to clone the repository and create a new environment using the 
environment.yml file. In a terminal (e.g. a miniconda/Anaconda prompt):

```bash
git clone https://git.embl.de/kbeckwit/looptrace
cd looptrace
conda env create -f environment.yml
python setup.py install
conda activate pychrtrace
```
There are two optional (large) package required if deconvolution and nuclei segmentation that can be used in the analysis. These can be installed using:
```bash
pip install flowdec[tf_gpu]
pip install cellpose
```

## Usage:
First edit the config YAML file to provide input and output directories and other parameters. An example file is found in example_config/example_config.yaml.

Once edited, to the pychrtrace directory and execute in a terminal:

```bash
python bin\tracing_gui.py
```
Follow instructions as required.

## Authors
Written and maintained by Kai Sandvold Beckwith (kai.beckwith@embl.de), Ellenberg group, CBB, EMBL Heidelberg.
See https://www-ellenberg.embl.de/. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
Please cite our paper.
