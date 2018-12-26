FROM spectrocat/python37:1

# install the development version of SpectroChemPy
# -------------------------------------------------

# clone the git SpectroChemPy repository
WORKDIR /usr/src/spectrochempy
RUN git clone https://fernandezc@bitbucket.org/spectrocat/spectrochempy.git .

RUN pip install --upgrade pip
RUN pip install -e .
RUN conda install nbconvert=5.3

CMD [ "/bin/bash" ]