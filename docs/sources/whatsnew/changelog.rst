
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

- ``ScpObjectList`` gains four convenience methods for selecting and
  filtering datasets when reading multi-object files (MATLAB ``.mat``,
  multi-subfile SPC, ZIP archives):

  - ``select_largest(ndim=None)`` — select the dataset with the most
    elements, optionally restricted to a given number of dimensions.
  - ``select_by_name(name)`` — select the first dataset whose name
    contains the given substring (case-insensitive).
  - ``filter_by_ndim(ndim)`` — return a new ``ScpObjectList`` with only
    datasets of the given dimensionality.
  - ``filter_by_shape(shape)`` — return a new ``ScpObjectList`` with only
    datasets matching the exact shape.

  A ``names`` property is also added to quickly inspect all dataset
  names. (#1306)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``PLSRegression`` now works with a 1D ``NDDataset`` as the response variable
  ``y`` (shape ``(n_obs,)``). Previously, the ``_set_output`` coordinate wrapping
  decorator assumed the metadata source was always 2D, causing a ``ValueError``
  on ``predict()``, ``y_scores``, ``y_loadings``, ``y_weights``, ``y_rotations``
  and ``result`` when fitting with a 1D target.  Fixes the ``coef`` property
  coordinate assignment which incorrectly used ``self._Y.x`` instead of
  ``self._Y.y`` for the target dimension. (#1305)


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
