# Choose the minimal jupyter image: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM jupyter/base-notebook:latest

USER root

RUN sudo apt-get update -y && \
    sudo apt-get install -y libx11-6

USER $NB_UID

# Choose python 3.8 version
ARG PY_VERSION=3.8
ARG DEV=''
ARG DASH=''
ARG CANTERA=''
ARG BRANCH=''

# ENV CONDA_ENV scpy$PY_VERSION

# We will first create a YAML file and then use it to update the environment
# to be suitable for spectrochempy

COPY --chown=${NB_UID}:${NB_GID} .ci/scripts/env* /home/$NB_USER/tmp/

RUN cd /home/$NB_USER/tmp/ && \
    conda update conda && \
    conda install -c conda-forge mamba jinja2 && \
    python env_create.py -v $PY_VERSION $DEV $DASH $CANTERA scpy$PY_VERSION.yml && \
    # mamba env create -p $CONDA_DIR/envs/$CONDA_ENV -f scpy$PY_VERSION.yml && \
    mamba env update --name base -f scpy$PY_VERSION.yml && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/tmp


# Install spectrochempy
COPY --chown=${NB_UID}:${NB_GID} . /home/$NB_USER/spectrochempy/

RUN cd spectrochempy && \
    git checkout $BRANCH && \
    python setup.py install
