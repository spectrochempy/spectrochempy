
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

- Split one dataset axis into blocks with ``reshape(coord_policy='split')``.
- Compute and invert one-sided Fourier transforms along a dataset dimension with ``rfft`` and ``irfft``.
- ``NDDataset.to_markdown_report()`` now renders a deterministic Markdown
  summary of one explicitly supplied dataset, selected metadata, structured
  history and explicit figure references. It does not capture Matplotlib state
  or claim exhaustive provenance. Every recorded history operation is included
  in order; only individual values are explicitly truncated. Paths are rendered
  as supplied, and figure references accept relative paths and HTTP(S) URLs. (:pr:`1775`)

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Preserve metadata lists whose length differs from the number of dataset
  dimensions when transposing or swapping axes after loading an SCP file. (:pr:`1885`)


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

- PERF: Speed up IRIS fits by preparing the OSQP matrix once per regularization
  parameter, resolving the backend once per fit, and reducing the size of the
  sharpness calculation for wide datasets. (:pr:`1887`)
