
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

- The welcome message display functionality has been removed.
- Implemented lazy matplotlib initialization: SpectroChemPy now loads matplotlib only when plotting is actually used, reducing import-time overhead for non-plotting workflows.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Refactored SpectroChemPy initialization and plotting setup to reduce Matplotlib side effects (issue #877). SpectroChemPy no longer modifies global ``matplotlib.rcParams``.
- Improved handling of Matplotlib styles, separating logical styles (e.g. ``default``) from file-based ``.mplstyle`` styles.
- Stabilized plot preferences reset and style application across tests, docs, and CI environments.
- Fixed scikit-learn ``PLSRegression`` compatibility with scikit-learn >= 1.5.
- Fixed issue #858 (OMNIC series reader): Added ``reverse_x`` parameter for wavenumber ordering.
- Fixed issue #856 (osqp dependency, used for IRIS): Added warning for osqp < 1.0.
- Fixed MCRALS: Corrected indexing for intermediate spectral matrices (C and St lists).
- Fixed issue #875: Cannot subtract offset-naive and offset-aware datetimes.



.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- Major Python compatibility updates:
    - Maximum Python version increased to 3.14
    - Minimum Python version increased to 3.11
    - Dropped support for Python 3.10 and below
- osqp > 1.0 is now allowed (#856). A warning is shown for osqp < 1.0, which will not be supported in future versions.


.. section

Breaking Changes
~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Dropped support for Python 3.10 and below.
- ``restore_rcparams()`` is now deprecated: SpectroChemPy no longer modifies global ``matplotlib.rcParams``, so restoration is unnecessary.
- Plotting initialization behavior has changed internally to reduce global Matplotlib side effects. While intended to be backward compatible, code relying on implicit global ``rcParams`` side effects may behave differently.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- ``IRIS.plotdistribution(index)`` is now deprecated. Use ``IRIS.f[index].plot()`` instead.
