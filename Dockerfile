FROM continuumio/miniconda3

RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y

# clone the git SpectroChemPy repository
RUN git clone https://fernandezc@bitbucket.org/spectrocat/spectrochempy.git /usr/src/spectrochempy

# Create a python 3.7 environment
RUN conda update -n base -c defaults conda
RUN conda env create -f /usr/src/spectrochempy/scp37env.yml
RUN source activate scp37

#RUN conda config --prepend channels conda-forge
#RUN conda install --yes --file requirements.txt

# install the development version of SpectroChemPy
RUN pip install -e /usr/src/spectrochempy


ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

