FROM continuumio/miniconda3:4.10.3
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

# Update and create base image.
RUN apt-get update -y &&\g
    apt-get install -y gcc g++ make libz-dev &&\
    apt-get clean

# Install mamba and the Python environment.
RUN conda install mamba -n base -c conda-forge

# Get the looptrace code.
RUN git clone https://github.com/gerlichlab/looptrace.git

# Set up looptrace.
RUN cd looptrace &&\
    mamba env update -n base --f environment.yml &&\
    mamba activate base &&\
    python setup.py install

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

# Finish setup.
USER jovian
ENV PATH=/opt/conda/bin:${PATH}
CMD ["/bin/bash"]

