FROM nvidia/cuda:12.3.2-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install build-essential software-properties-common -y && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt-get install git wget vim -y && \
    # apparently needed to make TensorFlow 2.16 communicate with our GPUs through CUDA with 12.3.2
    apt-get install cuda-toolkit-12-3 -y

# Copy repo code, to be built later.
RUN mkdir /looptrace
WORKDIR /looptrace
COPY . /looptrace
RUN mv /looptrace/target/scala-3.4.1/looptrace-assembly-0.5-SNAPSHOT.jar /looptrace/looptrace

# Install new-ish R and necessary packages.
RUN echo "Installing R..." && \
    /bin/bash /looptrace/setup_image/allow_new_R.sh && \
    apt-get install r-base -y && \
    R -e "install.packages(c('argparse', 'data.table', 'ggplot2', 'stringi'), dependencies=TRUE, repos='http://cran.rstudio.com/')" && \
    echo "Installed R!"

# Install minimal Java 21 runtime, in updates repo for Ubuntu 20 (focal) as of 2024-01-19.
RUN apt-get update -y && \
    apt-get install openjdk-21-jre-headless -y

# Install miniconda.
## The installation home should be /opt/conda; if not, we need -p /path/to/install/home
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# For the CUDA-based container, we only need to add the Python env (because we install TensorFlow there).
ENV PATH=/opt/conda/bin:${PATH}

# Build the looptrace package, with extra dependencies for pipeline.
# This group of extras should be declared in the pyproject.toml.
RUN pip install .[deconvolution,pipeline]

# Point to the CUDA libraries for integration with TensorFlow
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.3/targets/x86_64-linux/lib

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
