FROM debian:latest

LABEL maintainer = "spectrocat <https://bitbucket.org/spectrocat/spectrochempy>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get -qq update --fix-missing

RUN apt-get -qq -y install curl bzip2 libgl1-mesa-glx libglib2.0-0 ca-certificates libxext6 libsm6 libxrender1 git
RUN apt-get -qq -y --reinstall install libqt5dbus5 libqt5widgets5 libqt5network5 libqt5gui5 libqt5core5a libdouble-conversion1 libxcb-xinerama0
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh

RUN bash /tmp/miniconda.sh -bfp /opt/conda && rm -rf /tmp/miniconda.sh

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
 echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
 echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda
RUN conda config --append channels conda-forge
COPY requirements.txt  .
RUN cat requirements.txt
RUN conda install --yes --file requirements.txt
RUN rm requirements.txt

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD [ "/bin/bash" ]