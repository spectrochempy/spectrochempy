.. _install_colab:

****************************************
Install in Google Collaboratory (Colab)
****************************************

May be you want to try or run SpectroChemPy without installing python and all the necessary packages on your computer.

To do this you can use `Colaboratory <https://colab.research.google.com/notebooks/intro.ipynb?hl=en#>`__,
in short `Colab`, which allows you to run python notebooks in your browser without any installation.

The Colab Notebooks are very similar to Jupyter Notebook (from which they derive).

To start with Colab, go to the `Colab website <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`_
and create a new notebook.

In the first cell, you can enter the following block of instructions. This is rather verbose, so if you want to clear
automatically the output after execution, you can uncomment the last line.

.. sourcecode:: ipython3

    from IPython.display import clear_output
    !git clone https://github.com/spectrocat/spectrochempy.git
    !pip install -r ./spectrochempy/requirements.txt
    !pip install ./spectrochempy
    # clear_output()


Then as usual you can start using SpectroChemPy.

.. image:: images/colab.png
       :alt: Colab windows


.. warning::

   Colab notebooks are isolated and thus you need to perform the above operation for all notebook you create.
