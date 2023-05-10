FROM continuumio/miniconda3:latest
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

# Update and create base image.
# TODO: is each dependency here really needed?
RUN apt-get update -y &&\
    apt-get install -y gcc g++ make libz-dev &&\
    apt-get clean

RUN cd /opt && mkdir looptrace
WORKDIR /opt/looptrace
COPY . .

# Create conda env and install mamba, matching environment 
# name to that in the config file to be used. 
# Install also the looptrace dependencies.
# NB: The mamba default env name is evidently base.
RUN conda install mamba -n base -c conda-forge
RUN mamba env update -n base --file environment.yaml
RUN python setup.py install
RUN mamba list > software_versions_conda.txt

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
