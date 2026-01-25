:orphan:

What's New in Revision 0.8.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.8.2.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

Bug Fixes
~~~~~~~~~

- fixed scikitlearn PLSRegression compatibility with scikit-learn >= 1.5
- fixed issue #858 (omnic series reader)
- fixed issue #856 (osqp dependency, used for IRIS)
- fixed MCRALS (list od intermediate spectral matrices)
- omnic_reader properly reads units for single beam spectra
- fixed issue #875 (can't subtract offset-naive and offset-aware datetimes)

Dependency Updates
~~~~~~~~~~~~~~~~~~

* Major Python compatibility updates:
    - Maximum Python version increased to 3.14
    - Minimum Python version increased to 3.11
    - Dropped support for Python 3.10 and below
- osqp > 1.0 now allowed (#856). A warning has been added, osqp < 1.0 will not be supported in the future.

Breaking Changes
~~~~~~~~~~~~~~~~

- python 3.10 osqp > 1.0 now allowed (#856). A warning has been added, osqp < 1.0 will not be supported in the future.
