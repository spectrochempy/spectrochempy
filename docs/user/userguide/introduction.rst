.. _userguide.introduction

Introduction
############

The |scpy| project was developed to provide advanced tools for processing and
analyzing spectroscopic data, initially for internal purposes in the
`LCS <https://www.lcs.ensicaen.fr/>`_.

Scpy is essentially a library written in python language and which proposes objects (NDDataset, NDPanel and Project)
to contain data, equipped with methods to analyze, transform or display this data in a simple way by the user.

The processed data are mainly spectroscopic data from techniques such as IR, Raman or NMR, but they are not limited
to this type of application, as any type of numerical data arranged in tabular form can generally serve as the main
input.


How to get started
******************

**Note** We assume that the SpectroChemPy package has been correctly
installed. if is not the case, please go to `SpectroChemPy installation
procedure <reference/install/index.rst>`_.


.. toctree::
   :maxdepth: 2

   introduction/interface


Loading the API
****************

The |scpy| API exposes many objects and functions.

To use the API, you must import it using one of the following syntax:

.. ipython:: python

    import spectrochempy as scp
    nd = scp.NDDataset()

.. ipython:: python

    from spectrochempy import *
    nd = NDDataset()

With the second syntax, as often in python, the access to objects/functions
can be greatly simplified. For example, we can use "NDDataset" without a prefix
instead of `scp.NDDataset` which is the first syntax) but there is always a risk
of overwriting some variables or functions already present in the namespace.
Therefore, the first syntax is generally highly recommended.


* :ref:`Continue with the description of Spectrochempy's basic objects <userguide.objects>`
