FROM tensorflow/tensorflow:2.16.2-gpu
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8

RUN apt-get update -y && \
    apt-get install build-essential software-properties-common -y && \
    # Need to link apt_get.so: https://askubuntu.com/questions/1043484/modulenotfounderror-no-module-named-apt-pkg-trying-to-install-moka
    ln -s /usr/lib/python3/dist-packages/apt_pkg.cpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.so && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt-get install git wget vim -y

# Copy repo code, to be built later.
RUN mkdir /looptrace
WORKDIR /looptrace
COPY . /looptrace
RUN mv /looptrace/target/scala-3.5.2/looptrace-assembly-0.13.0-SNAPSHOT.jar /looptrace/looptrace

# Install new-ish R and necessary packages.
RUN echo "Installing R..." && \
    /bin/bash /looptrace/setup_image/allow_new_R.sh && \
    apt-get install r-base -y && \
    R -e "install.packages(c('argparse', 'data.table', 'ggplot2', 'stringi'), dependencies=TRUE, repos='http://cran.rstudio.com/')" && \
    echo "Installed R!"

# Install minimal Java 21 runtime, in updates repo for Ubuntu 20 (focal) as of 2024-01-19.
RUN apt-get update -y && \
    apt-get install openjdk-21-jre-headless -y

# Build the looptrace package, with extra dependencies for pipeline.
# This group of extras should be declared in the pyproject.toml.
RUN pip install .[deconvolution,pipeline]

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
