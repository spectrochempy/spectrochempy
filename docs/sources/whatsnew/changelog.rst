
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- DOC: Improved example gallery to showcase SpectroChemPy-native idioms
  (``Coord.linspace``, ``Coord.arange``, ``scp.abs``) for coordinate creation
   and dataset operations, replacing redundant ``np.linspace`` + ``Coord``
   wrapping patterns, ``np.abs`` usage, list-comprehension synthetic data
   generators (``scp.fromfunction``), ``np.random.normal`` on datasets
   (``scp.normal``), ``np.arange`` wrapped in NDDataset (``scp.arange``),
   and ``np.random.rand`` + NDDataset constructor (``NDDataset.random``).
   Also updated API docstring examples to use ``scp.gaussian``,
   ``Coord.linspace``, ``scp.arange``, and ``NDDataset.random``
   instead of raw NumPy equivalents. (#1370)
