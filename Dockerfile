FROM tensorflow/tensorflow:2.11.1-gpu
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install build-essential software-properties-common -y
 
# Install other build dependencies git and wget and zlib.
RUN apt-get install git wget libz-dev libbz2-dev liblzma-dev -y

# Clone repo.
RUN cd / && mkdir looptrace && cd /looptrace
WORKDIR /looptrace
COPY . .

# Install miniconda.
## The installation home should be /opt/conda; if not, we need -p /path/to/install/home
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:/usr/local/lib/python3.8/dist-packages:${PATH}

RUN echo "which conda 2" && \
    which conda

# Create conda env and install mamba, matching environment 
# name to that in the config file to be used. 
# Install also the looptrace dependencies.
# NB: The mamba default env name is evidently base.
RUN conda install mamba -n base -c conda-forge && \
    mamba env update -n base --file environment.yaml && \
    python setup.py install && \
    mamba list > software_versions_conda.txt

RUN cd /opt/conda/lib/python3.8/site-packages/tensorrt_libs && \
    ln -s libnvinfer.so.8 libnvinfer.so.7 && \
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

# Make relevant software available on the pertinent paths.
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3.8/dist-packages
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/python3.8/site-packages/tensorrt_libs

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

