
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

- PCA component labels now display as ``PC1``, ``PC2``, ... instead of ``#0``,
  ``#1``, ... in legends and coordinate display. Other analysis methods
  retain the default ``#0``, ``#1``, ... labels. (#1404)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Display legend in 2D lines/stack plots when ``legend=True`` is passed as a
  plotting keyword argument. Previously the ``legend`` kwarg was silently
  ignored by the 2D plot backend, so lines rendered with auto-populated labels
  from coordinate metadata never showed a legend. (#1404)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- ``MCRALS.solverConc`` and ``MCRALS.solverSpec`` are deprecated in favour
  of ``solver_C`` and ``solver_St``. The old names remain functional but emit
  :class:`FutureWarning`. (:pr:`XXXX`)

- ``MCRALS.constraints`` is now a validated traitlet, enabling both constructor
  and post-construction assignment while preserving the distinction between
  ``None`` (legacy path) and ``[]`` (explicitly unconstrained new-API fit).
  Assignment of ``constraints`` after fitting invalidates the fitted state.
  The ``constraints`` parameter is not config-file serializable. (:pr:`XXXX`)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- ``MCRALS.solverConc`` is deprecated; use ``solver_C`` instead.
- ``MCRALS.solverSpec`` is deprecated; use ``solver_St`` instead.

- Legacy MCR-ALS constraint traitlet parameters (``nonnegConc``,
  ``unimodConc``, ``closureConc``, etc.) are deprecated; use the
  ``constraints`` API instead.


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
