# Loop tracing in python.

LoopTrace is a Python package for chromatin tracing image and data analysis as described in https://doi.org/10.1101/2021.04.12.439407

## Installation

The simplest way to install the package and all dependencies is to clone the repository and create a new environment using the 
environment.yml file. There are several optional dependencies, these can be installed separately (see below).
In a terminal (e.g. a bash/miniconda/Anaconda prompt):

```bash
git clone https://git.embl.de/grp-ellenberg/looptrace
cd looptrace
conda env create -f environment.yml
conda activate looptrace
python setup.py install
```
There are several optional packages that can be added depending on use case:

Deconvolution of spots during tracing (depending on your environment you may wish to install or Tensorflow seperately):
```bash
pip install flowdec[tf_gpu]
```

## Basic usage (chromatin tracing):
See example notebooks (notebooks folder) for running the provided CLI scripts either locally or in an HPC environment. Each CLI script is also documented. Updated GUI documentation will be provided shortly.

Once the tracing is done, the data can be further analyzed for example using iPython notebooks. See examples of the full analysis as well as several full datasets at https://www.ebi.ac.uk/biostudies/studies/S-BIAD59

## Authors
Written and maintained by Kai Sandvold Beckwith (kai.beckwith@embl.de), Ellenberg group, CBB, EMBL Heidelberg.
See https://www-ellenberg.embl.de/. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
Please cite our paper: https://www.biorxiv.org/content/10.1101/2021.04.12.439407
