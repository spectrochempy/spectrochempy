
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

- Result objects now provide attribute-style access to named outputs and
  diagnostics, for example `pca.result.scores`, `pca.result.loadings`, and
  `pca.result.explained_variance`. The IRIS and TENSOR/CP plugins now follow
  the same contract through `iris.result` and `cp.result`.

- Added a minimal MATLAB `.mat` writer for simple numeric `NDDataset`
  exchange using `scipy.io.savemat`. This is an exchange format, not
  native SpectroChemPy persistence.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Standardized unsupported reader errors so invalid protocols, unrecognized
  file types, and unsupported CSV origins report the affected filename,
  requested or detected value, and supported alternatives. (#1143)

- Clarified specialized writer docstrings so format-specific APIs such as
  `write_csv()`, `write_jcamp()`, and `write_matlab()` no longer advertise the
  generic multi-protocol export options documented on `write()`.

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
