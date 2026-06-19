
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

- FIX: handle SVD ``compute_uv=False`` outputs consistently. The private
  ``_outfit`` attribute is now always a ``(U, s, VT)`` tuple, fixing wrong
  values and crashes for all properties when ``compute_uv=False``.
  (:pr:`1210`)

- FIX: pass the ``solver`` configuration parameter to the underlying
  ``sklearn.decomposition.NMF`` object. Previously the parameter was
  accepted by the SpectroChemPy ``NMF`` estimator but silently ignored
  during sklearn initialisation, so the default solver was always used
  regardless of the configured value. (:pr:`1212`)

- FIX: correct ``PLSRegression.n_iter`` property attribute name. Previously,
  the property referenced ``self._n_iter_`` instead of the actual stored
  attribute ``self._n_iter``, causing an ``AttributeError`` after a
  successful fit. (:pr:`1216`)

- Fixed several inconsistencies in the OMNIC SPA/SPG reader (`#1144
  <https://github.com/spectrochempy/spectrochempy/issues/1144>`_):
  ``read_spg()`` no longer advertises the ``spa`` protocol; swapped error
  messages in SPG unit-consistency checks now name the correct quantity;
  a single SPA comment is now included in the description (previously only
  ≥2 comments were shown); SPA history content is now checked via the
  variable instead of a literal string; and invalid ``return_ifg`` values
  now produce a clear warning. (PR by @gaoflow)

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

- MAINT: introduce the new Result Object architecture (``ResultBase``,
  ``AnalysisResult``, ``FitResult``) for analysis and fitting workflows.
  (:pr:`1208`)

- MAINT: extend Result Object support to PCA, SVD, NMF, Optimize,
  and MCRALS, providing unified access to analysis and fitting outputs
  while preserving backward compatibility. (:pr:`1208`, :pr:`1209`,
  :pr:`1211`, :pr:`1213`, :pr:`1215`)
