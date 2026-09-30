
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

- ``output`` is now honored by the plotting API. ``dataset.plot()``,
  ``plot_multiple()``, ``multiplot()``, and the composite plotters
  (``plot_score()``, ``plot_scree()``, ``plot_compare()``, ``plot_merit()``,
  ``plot_baseline()``, and ``plot_parity()``) write the finished figure to the
  requested file, accepting both ``str`` and ``pathlib.Path``. The whole figure
  is saved once the plot is complete - legend, colorbars, and multi-panel
  layouts included - and before the display step, so the file is on disk even
  when ``show=True`` keeps a window open. The file format follows the
  extension, the existing ``savefig.*`` preferences drive resolution,
  background, and bounding box, and a missing parent directory is reported as
  an ``OSError`` instead of being created silently. ``multiplot()`` also
  performs a single display step for the whole grid instead of one per panel
  (:pr:`1717`).


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
