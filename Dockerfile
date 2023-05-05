FROM continuumio/miniconda3:4.10.3
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

# Update and create base image.
# TODO: is each dependency here really needed?
RUN apt-get update -y &&\
    apt-get install -y gcc g++ make libz-dev &&\
    apt-get clean

# Get and setup looptrace code in a dedicated conda environment
RUN git clone https://github.com/gerlichlab/looptrace.git
RUN cd looptrace && conda env create -f environment.yml
SHELL ["conda", "run", "-n", "looptrace", "/bin/bash", "-c"]
RUN cd looptrace && python setup.py install

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

