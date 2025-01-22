.. _install_colab:

****************************************
Install in Google Colaboratory (Colab)
****************************************

Maybe you want to try or run SpectroChemPy without installing python and all the
necessary packages on your computer.

To do this you can use
`Colaboratory <https://colab.research.google.com/notebooks/intro.ipynb?hl=en#>`__,
in short `Colab`, which allows you to run python notebooks in your browser without any
installation.

The Colab Notebooks are very similar to Jupyter Notebook (from which they derive).

To start with Colab, go to the
`Colab website <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`__
and create a new notebook or simply execute the `example notebook on Colab <https://colab.research.google.com/github/spectrochempy/spectrochempy_tutorials/blob/main/colab/spectrochempy_on_colaboratory.ipynb>`__  |Open in Colab|

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/spectrochempy/spectrochempy_tutorials/blob/main/colab/spectrochempy_on_colaboratory.ipynb

Load and install SpectroChemPy with pip
---------------------------------------

In the first cell, enter and execute the following block of instructions to load and
install SpectroChemPy in Colab.

.. sourcecode:: ipython3

    ! pip install -q spectrochempy &>> output.log

The whole process takes less than 1 minute. Then as usual in Jupyter Notebook you can start using SpectroChemPy by inserting
adding new code cells:

.. image:: images/colab_1.png
       :alt: Colab windows

.. warning::

   Colab notebooks are isolated and thus you need to perform the above operation
   for all notebook you create... and to run it each time you reopen the notebook
   for further processing or analysis.

Load data files
---------------
As Colab notebooks are hosted on Google servers, they have no direct access to your
local drive (unlike a Jupyter notebook for instance).
There are several ways to load datafiles on the server:

* Through the file-explorer pane located on the left-hand side of the Colab window
  (NOT the File menu which is devoted to notebooks and scripts management !),

* Equivalently by importing the `files` module from the `google.colab` library and
  runing the `files.upload()` widget (works best on Google Chrome).

.. sourcecode:: ipython3

    from google.colab import files
    files.upload()

* By connecting the notebook to your Google Drive (this of course requires that your
  local files are synchronized with Google drive), using the file-explorer pane.

* By cloning a GitHub repository using `git clone` .

.. image:: images/colab_2.png
       :alt: Colab windows
