
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

- Elementwise arithmetic on ``NDDataset``/``NDArray`` now follows a single
  shared title rule engine, making Python operators and their ``numpy`` ufunc
  counterparts produce identical titles. Identity-preserving operations
  (``abs``, ``negative``, ``positive``, rounding, ...) and dataset-scalar
  additive or dimensionless-scaling operations keep the source ``title``
  verbatim; unary transforms compose it (``sin(source)``, ``sqrt(source)``);
  dataset-dataset operations compose ``add(...)``/``subtract(...)``/
  ``multiply(...)``/``divide(...)``; powers compose ``power(source, p)``. The
  previous ``"<fname>"`` substitution and the operator/ufunc title divergence
  have been removed, and composed titles longer than 120 code points collapse
  to an absent title.

- Derived analysis outputs now remain compatible with source datasets whose
  metadata contains read-only nested structures. In particular,
  ``Optimize.predict()`` and ``Optimize.result.residuals`` now attach detached
  writable metadata copies instead of attempting in-place updates against
  locked source-derived metadata.

- The AsLS baseline-correction path now avoids the SciPy
  ``SparseEfficiencyWarning`` previously exposed by published gallery examples,
  without changing the underlying baseline-correction algorithm or the
  scientific results.

- Fix inconsistent fit diagnostics after directly mutating the public
  ``Optimize.fp`` view (for example setting ``fp.fixed``): the mutation was
  ignored by the optimization but was re-injected into the reported state by
  the post-fit script round-trip, so ``n_varying_parameters``,
  ``degrees_of_freedom`` and the rendered script could disagree with what was
  actually optimized. The canonical model spec is now the sole source of truth
  after a fit.


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

- ``Optimize`` no longer rebuilds its canonical model state by re-parsing the
  post-fit rendered script. After a successful fit the canonical spec keeps the
  full-precision optimized values, the public ``Optimize.fp`` view keeps its
  identity and its in-place synced values, and ``Optimize.script`` becomes a
  rendered representation of the fitted values: the display precision of the
  render no longer limits the precision of the internal fitted state, and the
  rendered text is never re-injected into the canonical model.
  ``FitParameters`` and ``Optimize.fp`` are unchanged, the fp-only entry path is
  preserved, and no API is removed or deprecated.

- The ``Optimize`` structured-validation flow now validates constraint
  parameter-name references against the canonical ``_FitModelSpec``
  representation instead of the legacy ``FitParameters`` view (``Optimize.fp``).
  ``FitParameters`` and ``Optimize.fp`` remain available and unchanged, and the
  fitting DSL and scientific results are unaffected.
