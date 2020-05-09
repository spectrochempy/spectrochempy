FROM continuumio/miniconda3

RUN conda update -n base -c defaults conda

# Create the scpy environment
ADD scpy.yml /tmp/scpy.yml
RUN conda env update -n scpy -f /tmp/scpy.yml

# activate this environment
ENV PATH /opt/conda/envs/scpy/bin:$PATH
RUN echo "source activate scpy" > ~/.bashrc


