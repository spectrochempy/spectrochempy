.. currentmodule:: spectrochempy

.. _glossary:

********
Glossary
********

.. glossary::
    :sorted:

    API
        API stands for ``Application Programming Interface``, which is a set of methods
        and protocols for using the SpectroChemPy (especially
        in `Jupyter Notebooks <https://docs.jupyter.org/en/latest/>`__
        or `Jupyter Lab <https://docs.jupyter.org/en/latest/>`__\ )
        without knowing all the details of the implementation of these methods or protocols.

    Carroucell
        Multisample FTIR cell.

    SVD
        ``Singular Value Decomposition``\ .

    PCA
        ``Principal Component Analysis``\ .

    EFA
        ``Evolving Factor Analysis``\ .

    MCR-ALS
        ``Multivariate Curve Resolution Alternating Least Squares``
        resolve's a set of spectra :math:`X` of an evolving mixture
        into the spectra :math:`S` of "pure" species and their
        concentration profiles :math:`C`\ , such as:

        .. math:: X = C . S^T

    ALS
        ``Alternating least squares`` minimization.
        *Describe this*

    closure
        Constraints

    unimodality
        *Describe this*

    regularization
        *Describe this*

    array-like
        An object which can be transformed into a 1D dataset such as a list or tuple of numbers, or a
        2D or nD dataset such a list of lists or a `~numpy.ndarray`\ .

    n_observations
        Number of ``observations``. When dealing with spectroscopic data, an ``observation``
        is generally a single spectra record.

    n_features
        Number of ``features``. A feature for a spectroscopic ``observation`` (spectra) is generally a measurement
        at a single frequency/energy or any derived quantity.

    n_components
        Number of underlying components for spectroscopic data.
