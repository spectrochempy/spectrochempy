
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

- the welcome message display functionality has been removed
- added `restore_rcparams()` to restore Matplotlib `rcParams` after SpectroChemPy plotting (issue #877).
- implemented lazy matplotlib initialization: SpectroChemPy now loads matplotlib only when plotting is actually used, providing dramatic performance improvements for non-plotting workflows (46% faster import times).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Performance improvement Implemented comprehensive lazy initialization system for matplotlib, eliminating import-time overhead while maintaining full functionality. Import time reduced from 232ms to 126ms (46% faster) with complete matplotlib isolation until first plot call.
- Refactored SpectroChemPy initialization and plotting setup to reduce Matplotlib side effects (issue #877).
- Improved handling of Matplotlib styles, separating logical styles (e.g. `default`) from file-based `.mplstyle` styles.
- Stabilized plot preferences reset and style application across tests, docs, and CI environments.
- fixed scikitlearn PLSRegression compatibility with scikit-learn >= 1.5
- fixed issue #858 (omnic series reader)
- fixed issue #856 (osqp dependency, used for IRIS)
- fixed MCRALS (list of intermediate spectral matrices)
- omnic_reader properly reads units for single beam spectra
- fixed issue #875 (can't subtract offset-naive and offset-aware datetimes)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- Major Python compatibility updates:
    - Maximum Python version increased to 3.14
    - Minimum Python version increased to 3.11
    - Dropped support for Python 3.10 and below
- osqp > 1.0 now allowed (#856). A warning has been added, osqp < 1.0 will not be supported in the future.

.. section

Breaking Changes
~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- python 3.10 osqp > 1.0 now allowed (#856). A warning has been added, osqp < 1.0 will not be supported in the future.
- No breaking changes on plotting: The lazy matplotlib initialization is fully backward compatible. All existing code continues to work without modification while benefiting from the performance improvements.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- IRIS.plotdistribution(index) is now deprecated.  Usee IRIS.f[index].plot() instead. A deprecation notice has been added.
