:orphan:

.. _contributing.docker:

**********************
Building Docker images
**********************

Docker is an open platform for developing, shipping, and running applications. It allows to develop and run
`SpectroChemPy` in an unified environment whatever the host platform is. For the end-user it should also help to
avoid all installation problems as the images created have been tested before being shipped.

If you like to understand better how it works you can go to the
`Docker tutorial <https://www.docker.com/101-tutorial>`__.

Docker Installation
====================

see :ref:`install_docker_details`

Building the environment image locally
=======================================

To build a `SpectroChemPy` image you can do it using the ``Docker build`` command in the directory where is located the
``Dockerfile`` :

.. sourcecode:: bash

   cd /path/to/spectrochempy/local/repository/.ci
   docker build -t spectrocat/spectrochempy:latest .

Due to the large number of operations to achieve (mainly downloading required packeges), this process can be rather
long the first time it is executed.
Further build after modification of the Dockerfile will generally be faster as most of the step are in a cache.

Creating and executing a Docker container
=========================================

Creating a container can be done using the Docker run command with the previously created image.

The recommended way:

.. sourcecode:: bash

   docker run -v /full/path/on/host/spectrochempy:/home/jovyan/spectrochempy \
              -p 8888:8888 \
              --name scpy \
              spectrocat/spectrochempy:latest \
              start.sh jupyter lab

This start a Jupyter Lab server on port 8888.

docker run    -p 8888:8888 \
              --name scpy \
              spectrocat/spectrochempy:latest \
              start.sh jupyter lab
