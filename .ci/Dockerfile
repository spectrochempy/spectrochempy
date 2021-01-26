# Choose the minimal jupyter image: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM jupyter/minimal-notebook:latest

# Choose python 3.9 version
ARG py_ver=3.9
ENV CONDA_ENV scpy${py_ver}

# Use mamba for faster building
RUN conda update conda && \
    conda install -c conda-forge mamba

# We will use a YAML file present in the docker build context to create the environment
COPY --chown=${NB_UID}:${NB_GID} env/scpy${py_ver}.yml /home/$NB_USER/tmp/
RUN cd /home/$NB_USER/tmp/ && \
    mamba env create -p $CONDA_DIR/envs/$CONDA_ENV -f scpy${py_ver}.yml && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/tmp

## Link it to jupyter
RUN $CONDA_DIR/envs/$CONDA_ENV/bin/python -m ipykernel install --user --name=$CONDA_ENV && \
   fix-permissions $CONDA_DIR && \
   fix-permissions /home/$NB_USER

# prepend conda environment to path
ENV PATH $CONDA_DIR/envs/$CONDA_ENV/bin:$PATH

# Make this environment to be the default one
ENV CONDA_DEFAULT_ENV $CONDA_DIR/envs/$CONDA_ENV

# Install spectrochempy
RUN python -m pip install git+https://github.com/spectrochempy/spectrochempy.git

WORKDIR /home/$NB_USER/work
