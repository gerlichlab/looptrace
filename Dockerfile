FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

# Include the cuda-compat-11-4 version to match what's on whatever machine for nvidia driver. Check nvidia-smi output.
# Include vim to facilitate editing the package and other files when in development
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install build-essential software-properties-common -y && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt-get install gcc-9 g++-9 -y && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --config gcc && \
    apt-get install git wget libz-dev libbz2-dev liblzma-dev -y && \
    apt-get install cuda-compat-11-4 -y && \
    apt-get install r-base -y && \
    R -e "install.packages(c('argparse', 'data.table', 'ggplot2', 'reshape2', 'stringi'), dependencies=TRUE, repos='http://cran.rstudio.com/')" && \
    apt-get install vim -y

# Copy repo code, to be built later.
RUN mkdir /looptrace
WORKDIR /looptrace
COPY . /looptrace
RUN mv /looptrace/target/scala-3.3.0/looptrace-assembly-0.1.0-SNAPSHOT.jar /looptrace/looptrace

# Install miniconda.
## The installation home should be /opt/conda; if not, we need -p /path/to/install/home
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# For the CUDA-based container, we only need to add the Python env (because we install TensorFlow there).
ENV PATH=/opt/conda/bin:${PATH}

# Build the looptrace package, with extra dependencies for pipeline.
# This group of extras should be declared in the pyproject.toml.
RUN pip install .[pipeline]

RUN cd /opt/conda/lib/python3.10/site-packages/tensorrt_libs && \
    ln -s libnvinfer.so.8 libnvinfer.so.7 && \
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

# For the CUDA-based container, we only need to add the tensorrt libraries path.
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/tensorrt_libs:/usr/local/cuda-11.4/compat:${LD_LIBRARY_PATH}

# Establish the current experiment data mount point, for convenient config file match and path operations.
ENV CURR_EXP_HOME=/home/experiment

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

# For Jupyter
CMD ["/bin/bash"]
