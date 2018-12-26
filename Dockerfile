FROM continuumio/miniconda3

# Add somme necessary librairies
RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y

# Create a python 3.7 environment with all required packages for spectrochempy

RUN conda config --prepend channels conda-forge
RUN conda update -n base -c defaults conda
RUN conda install --yes python=3.7.1
RUN curl -o requirements.txt  https://bitbucket.org/spectrocat/spectrochempy/downloads/requirements.txt
RUN conda install --yes --file requirements.txt

# install the development version of SpectroChemPy
# -------------------------------------------------

# clone the git SpectroChemPy repository
RUN git clone https://fernandezc@bitbucket.org/spectrocat/spectrochempy.git /usr/src/spectrochempy

RUN pip install --upgrade pip
RUN pip install -e /usr/src/spectrochempy

CMD [ "/bin/bash" ]

