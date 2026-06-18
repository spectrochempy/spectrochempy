
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
  regardless of the configured value. (:pr:`TODO`)


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

- MAINT: introduce ``ResultBase``, ``AnalysisResult``, and ``FitResult`` result
  object infrastructure as defined in the Result Object Contract RFC. A PCA-only
  prototype exposes ``pca.result`` as an ``AnalysisResult`` with named outputs
  (scores, loadings, components) and diagnostics (explained variance metrics)
  while preserving all existing public API and behavior. (:pr:`1208`)

- MAINT: extend ``AnalysisResult`` support to the SVD estimator. The new
  ``svd.result`` property exposes raw outputs (``U``, ``s``, ``VT``) and
  diagnostics (``singular_values``, ``explained_variance``,
  ``explained_variance_ratio``) while preserving all existing public API
  and backward compatibility. (:pr:`1211`)

- MAINT: introduce ``FitResult`` prototype for the ``Optimize`` estimator.
  The new ``opt.result`` property exposes fitted model outputs
  (``fitted``, ``components``) and solver diagnostics (``cost``,
  ``niter``, ``ncalls``) while preserving all existing public API and
  backward compatibility. (:pr:`1211`)

- MAINT: extend ``AnalysisResult`` support to the NMF estimator. The new
  ``nmf.result`` property exposes outputs (``components``, ``W``) and
  diagnostics (``reconstruction_error``, ``n_iter``) while preserving
  all existing public API and backward compatibility. (:pr:`TODO`)
