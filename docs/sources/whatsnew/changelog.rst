
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

- Added grouping and apparent axial widths for detected 2D peaks.

- Allow the independently released NMR 0.3 plugin in the ``nmr`` extra and
  link to its standalone reading and processing guides.

- Added ``integrate_region`` for physical ND boxes with optional propagated
  standard uncertainty on regular or irregular spectral grids.
- Added bounded multiplet fitting on multidimensional spectral grids to
  ``Optimize``, with fitted components, residuals and parameter uncertainties.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Preserve uniform frequency bins for FFTs with finely spaced axes so the
  inverse transform can recover the time-domain signal.
- Resolve physical phase pivots to sample indices before applying a
  first-order phase correction.

.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- ``fft()`` now leaves phase correction to an explicit
  ``pk()`` call. Pass the desired phase angles and pivot to ``pk()``.


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
