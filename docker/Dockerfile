FROM continuumio/miniconda3

# Create the scpy environment
ADD ../env/scpy.yml /tmp/scpy.yml
RUN conda env create -f /tmp/scpy.yml

# activate this environment
RUN conda activate scpy
