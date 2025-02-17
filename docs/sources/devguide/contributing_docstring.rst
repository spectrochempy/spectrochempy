:orphan:

.. _docstring:

SpectroChemPy Docstring Guide
=============================

This guide explains how to write docstrings for SpectroChemPy. We follow the NumPy docstring convention with some SpectroChemPy-specific additions.

.. contents:: Contents
   :local:
   :depth: 2

Docstring Format
----------------

SpectroChemPy uses `numpydoc <https://numpydoc.readthedocs.io/>`_ format for docstrings. Each docstring should contain:

1. A short summary (one line)
2. An extended description (optional)
3. Parameters section
4. Returns/Yields section
5. See Also section (optional)
6. Notes section (optional)
7. Examples section

Basic Rules
-----------

- Use triple double quotes ``"""``
- Start immediately after the quotes
- No blank lines before/after docstring
- Use backticks for code: ``parameter_name``, ``NDDataset``
- Link to classes/methods using :class:, :meth:, :func:
- End sections with periods

Example Structure
-----------------

.. code-block:: python

    def baseline(self, method="rubberband", **kwargs):
        """
        Apply baseline correction to the dataset.

        Corrects baseline using the specified method.

        Parameters
        ----------
        method : {"rubberband", "polyfit", "asls"}, default "rubberband"
            Method to use for baseline correction.
        **kwargs
            Additional parameters passed to the correction method.

        Returns
        -------
        NDDataset
            Dataset with corrected baseline.

        See Also
        --------
        detrend : Remove linear trends.
        smooth : Apply smoothing filters.

        Examples
        --------
        >>> ds = scp.read("myfile.spg")
        >>> ds_corrected = ds.baseline()
        """
        // implementation

Parameter Documentation
-----------------------

Types should be:

- Built-in types: ``int``, ``float``, ``str``, ``bool``
- Container types: ``list of int``, ``dict of {str: float}``
- SpectroChemPy types: ``NDDataset``, ``Coord``
- NumPy types: ``numpy.ndarray``, ``array-like``
- Multiple types: ``int or float``, ``str or None``

For choices, list options in curly braces:

.. code-block:: python

    method : {"linear", "polynomial"}, default "linear"
    axis : {0, 1, None}, default None

Examples Section
----------------

Examples should be:

1. Complete and runnable
2. As concise as possible
3. Show typical usage first
4. Written as doctest snippets

Always import SpectroChemPy as:

.. code-block:: python

    >>> import spectrochempy as scp

For plots, use the plot directive:

.. code-block:: python

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> ds = scp.read("myfile.spg")
        >>> ds.plot()

See the `numpydoc guide <https://numpydoc.readthedocs.io/>`_ for more details.
