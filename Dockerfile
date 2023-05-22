FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3
#FROM continuumio/miniconda3:latest
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

# Update the base image.
RUN apt-get update -y && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda.
## The installation home should be /opt/conda; if not, we need -p /path/to/install/home
## The -b option to the Miniconda installer provides "say yes" mechanism like -y for apt-get.
## The Python 3.8 version of the installer is needed as this is what the base container uses.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b
ENV PATH=/opt/conda/bin:${PATH}

# Copy this repo's code.
RUN cd /opt && mkdir looptrace
WORKDIR /opt/looptrace
COPY . .

# Create conda env and install mamba, matching environment 
# name to that in the config file to be used. 
# Install also the looptrace dependencies.
# NB: The mamba default env name is evidently base.
RUN conda install mamba -n base -c conda-forge && \
    mamba env update -n base --file environment.yaml && \
    python setup.py install && \
    mamba list > software_versions_conda.txt

# Reset working directory
WORKDIR /home

# User for VBC Jupyter Hub
ENV NB_USER jovian
ENV NB_UID 1000
ENV HOME /home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Set a notebook user.
USER jovian

# Make relevant software available on the pertinent paths.
## Ensure that Python and conda are available.
## Add the result of callig module load build-env/f2022 (on which CUDA 11.8.0 depends on our SLURM) and module load cuda/11.8.0.
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/software/f2022/software/cuda/11.8.0/nvvm/lib64:/software/f2022/software/cuda/11.8.0/extras/CUPTI/lib64:/software/f2022/software/cuda/11.8.0/

# Start bash.
CMD ["/bin/bash"]
