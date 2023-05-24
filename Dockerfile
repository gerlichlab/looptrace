FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install build-essential software-properties-common -y && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt-get install gcc-9 g++-9 -y && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --config gcc

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

ENV PATH=/opt/conda/bin:${PATH}

RUN echo "which conda 2" && \
    which conda

# Create conda env and install mamba, matching environment 
# name to that in the config file to be used. 
# Install also the looptrace dependencies.
# NB: The mamba default env name is evidently base.
RUN conda install mamba -n base -c conda-forge && \
    mamba env update -n base --file environment.yaml && \
    source activate base &&\
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
