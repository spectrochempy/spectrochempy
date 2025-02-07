.. currentmodule:: spectrochempy

.. _glossary:

********
Glossary
********

.. glossary::
    :sorted:

    ALS
        ALS stands for ``Alternating least squares`` minimization.
        The algorithm at the heart of term:`MCR-ALS` which successively resolves :math:`C` and :math:`St`
        by least squares, after application of the relevant constraints. It checks how
        :math:`\hat{X} = C \cdot St` is close to :math:`X` and either stops or goes for a new loop.

    API
        API stands for ``Application Programming Interface``, which is a set of methods
        and protocols for using the SpectroChemPy (especially
        in `Jupyter Notebooks <https://docs.jupyter.org/en/latest/>`__
        or `Jupyter Lab <https://docs.jupyter.org/en/latest/>`__)
        without knowing all the details of the implementation of these methods or protocols.

    AsLS
        AsLS stands for ``Asymmetric Least Squares`` smoothing.
        This method uses a smoother with asymmetric deviation weighting to obtain a baseline estimator.
        In doing so, this processor is able to quickly establish and correct a baseline
        while retaining information on the signal peaks.

    array-like
        An object which can be transformed into a 1D dataset such as a list or tuple of numbers, or a
        2D or nD dataset such a list of lists or a `~numpy.ndarray`.

    Carroucell
        Multisample FTIR cell as described in :cite:t:`zholobenko:2020`.

    closure
    closures
        Constraint where the sum of concentrations is fixed to a target value.

    EFA
        EFA stands for ``Evolving Factor Analysis``.
        EFA examines the evolution of the singular values or :term:`rank` of a dataset :math:`X` by systematically
        carrying out a :term:`PCA` of submatrices of :math:`X`. It is often used to guess predminance regions of
        appearing/disappearing species in an evolving mixture. See :cite:`maeder:1986` for the original case study
        and :cite:`maeder:2009` for more recent references.

    ICA
        ``Independant Component Analysis``.
        ICA is a method for separating a multivariate signal into additive subcomponents.

    loading
    loadings
        In the context of :term:`PCA`, loadings are vectors :math:`\mathbf{p}_i` of length :term:`n_features`
        which, associated to the corresponding :term:`score` vectors, are related to the so-called
        i-th principal component describing the variance of a datastet :math:`X`.

    MCR-ALS
        MCR-ALS stands for ``Multivariate Curve Resolution by Alternating Least Squares`` .
        MCR-ALS resolve's a set of spectra :math:`X` of an evolving mixture
        into the spectral profiles  :math:`S` of "pure" species and their
        concentration profiles :math:`C`, such as:

        .. math:: X = C \cdot S^T + E

        subjected to various soft constraints (such as non-negativity, unimodality, closure ...) or
        hard constraints (e.g. equality of concention(s) or of some spectra  to given profiles).

    n_components
        Number of underlying components or latent variables for spectroscopic data.

    n_features
        Number of ``features``. A feature for a spectroscopic ``observation`` (spectra) is generally a measurement
        at a single frequency/energy or any derived quantity.

    n_observations
        Number of ``observations``. When dealing with spectroscopic data, an ``observation``
        is generally a single spectra record.

    n_targets
        Number of ``targets``. A target is a property to predict using cross-decomposition methods such as PLS.
        Typically a target is a composition variable such as a concentration.

    NMF
        NMF stands for ``Non-negative Matrix Factorization``.
        NMF is a method for factorizing a non-negative matrix :math:`X` into two non-negative matrices
        :math:`W` and :math:`H` such as :math:`X = W \cdot H`. NMF is often used for feature extraction
        and dimensionality reduction.

    PCA
        ``Principal Component Analysis``.
        PCA is directly related to the :term:`SVD`. Its master equation is:

        .. math:: \mathbf{X} = \mathbf{T} \mathbf{P}^t + \mathbf{E}

        where :math:`\mathbf{T} \equiv U \Sigma` is called the :term:`scores` matrix and
        :math:`\mathbf{P}^t \equiv \mathbf{V}^t` the
        :term:`loadings` matrix. The columns of :math:`\mathbf{T}` are called the score vectors and the lines of
        :math:`\mathbf{P}^t` are called loading vectors. Together, the n-th score and loading vectors are related to
        a *latent variable* called the n-th principal component.

        Hence, :math:`\mathbf{T}` and :math:`\mathbf{P}` can then be viewed as collections of :math:`n` and :math:`m`
        vectors in k-dimensional spaces in which each observation/spectrum or feature/wavelength can be located.

        i-th principal component describing the variance of a datastet :math:`X`.

    PLS
        ``Partial Least Squares`` regression (or Projection on Latent Structures) is a statistical method to
        estimate :math:`n \times l` dependant or predicted variables :math:`Y` from :math:`n \times m`
        explanatory or observed variables :math:`X` by projecting both of them on new spaces spanned by
        :math:`k` latent variables according to the master equations :

        .. math::  X = S_X L_X^T + E_X

        .. math::  Y = S_Y L_Y^T + E_Y

        .. math::  S_X, S_y = \textrm{argmax}_{S_X, S_Y}(\textrm{cov}(S_X, S_Y))

        :math:`S_X` and :math:`S_Y` are :math:`n \times k` matrices often called X- and Y-score matrices, and :math:`L_X^T`
        and :math:`L_Y^T` are, respectively, :math:`k \times l` and :math:`k \times m` X- and Y-loading matrices.
        Matrices :math:`E_X` and :math:`E_Y`  are the error terms or residuals.
        As indicated by the third equation, the decompositions of :math:`X` and :math:`Y` are made to maximise
        the covariance of the score matrices.

    rank
        Number of linearly independent number or columns of a matrix

    regularization
        Technique used to reduce the errors of over-fitting a function on given data
        by adding, e.g. a smoothness constraint which extent is tuned by a regularization
        parameter :math:`\lambda` .

    score
    scores
        In the context of :term:`PCA`, scores are vectors :math:`\mathbf{t}_i` of length :term:`n_observations`
        which, associated to the corresponding :term:`loading` vectors, are related to the so-called


    SVD
        SVD stands for ``Singular Value Decomposition``.
        SVD decomposes a matrix :math:`\mathbf{X}(n,m)` (typically of set of :math:`n` spectra) as:

        .. math:: \mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^t + \mathbf{E}

        where :math:`\mathbf{U}(n,k)` and :math:`\mathbf{V}^t(k,m)` are matrices regrouping so-called left
        and right singular vectors of size :math:`k \leq \min(n,m)`. The factorization is exact (null
        error :math:`E`) whan :math:`k = \min(n,m)`. Among other properties, left and right singular
        vectors form two orthonormal basis of :math:`k`-dimensional spaces.
        Hence, for :math:`\mathbf{U}`:

        .. math:: \mathbf{u}_i\mathbf{u}_j^t = \delta_{ij}

        :math:`\Sigma` is a diagonal :math:`k\times k` matrix which diagonal elements :math:`\sigma_i`
        are called the  *singular values* of the matrix :math:`X`. The number :math:`r` of non-negligible
        (formally non-null) sigular values is called the :term:`rank` of :math:`X` and determines the
        number of linear independent lines or columns of :math:`X`.

        The singular values :math:`\sigma_i` are generally chosen in descending order, so that the first
        component -  :math:`\sigma_1 \mathbf{u}_1\mathbf{v}_1^t` models most of the dataset :math:`\mathbf{X}`, the second
        component models most of the remaining part of :math:`\mathbf{X}`, etc... Overall, the dataset
        can thus be reconstructed by the sum of the first :math:`r` components:

        .. math:: \mathbf{X} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i\mathbf{u}_j^t

        Finally, the sum of these singular values is equal to the total variance of the spectra and each singular
        value represents the amount of variance captured by each component:

        .. math:: \% \textrm{variance explained} = \frac{\sigma_i}{\sum_{i=  1}^r \sigma_i} \times 100

    unimodality
        Constraint where the profile has a single maximum or minimum.
