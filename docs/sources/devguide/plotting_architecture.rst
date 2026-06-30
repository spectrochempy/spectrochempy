.. _plotting-architecture:

Plotting Architecture
=====================

This note summarizes the current plotting architecture for contributors.  It is
descriptive, not normative: it explains where responsibilities live today so
that small plotting changes stay focused.

Public Entry Points
-------------------

The main public plotting entry points are:

* ``NDDataset.plot()`` for automatic dispatch based on dimensionality;
* explicit helpers such as ``plot_pen()``, ``plot_scatter()``,
  ``plot_bar()``, ``plot_lines()``, ``plot_contour()``,
  ``plot_contourf()``, ``plot_image()``, ``plot_surface()``, and
  ``plot_waterfall()``;
* orchestration helpers such as ``plot_multiple()`` and ``multiplot()``;
* composite plotters (for example PCA score/scree plots), which have their own
  specialized contracts.

Core Dispatch Path
------------------

The standard dataset plotting path is:

.. code-block:: text

    NDDataset.plot()
        -> plotting.dispatcher.plot_dataset()
        -> plotting.backends.matplotlib_backend.plot_dataset_impl()
        -> plot1d.py / plot2d.py / plot3d.py

The dispatcher selects the plotting backend.  The Matplotlib backend owns the
main dataset plotting dispatch and the final ``show`` step for that path.

Responsibility Split
--------------------

The current split is intentionally simple:

* ``plotting._methods``:
  normalize plotting vocabulary and compatibility aliases such as
  ``stack -> lines`` and ``map -> contour``.
* ``plotting._kwargs``:
  normalize plotting keyword aliases such as ``lw -> linewidth``,
  ``ls -> linestyle``, ``ms -> markersize``, ``mew -> markeredgewidth``,
  ``c -> color``, and ``colormap -> cmap``.
* ``plotting._style``:
  interpret normalized style inputs in a geometry-aware way
  (markers, line styles, palettes, colormaps, alpha, etc.).
* ``plot1d.py`` / ``plot2d.py`` / ``plot3d.py``:
  create Matplotlib artists and apply axis-level policy for the relevant
  plotting family.
* ``plot_multiple()``:
  overlay several datasets on one axes.
* ``multiplot()``:
  create a grid of axes and delegate per-panel rendering to the regular
  dataset plotting path.

Figure and Axes Lifecycle
-------------------------

Figure and axes ownership is partly centralized and partly specialized:

* the main 1D/2D dataset plotting path uses ``NDDataset._figure_setup()`` to
  decide whether to create a new figure, reuse the current figure, or reuse an
  explicitly provided ``ax``;
* ``ax``, ``clear``, and ``show`` are the main lifecycle controls for ordinary
  dataset plots;
* ``plot_multiple()`` and ``multiplot()`` orchestrate figures and axes
  themselves because they compose several plot calls;
* composite plots may own their figure lifecycle directly rather than going
  through the standard dataset path.

Practical Contributor Guidance
------------------------------

When changing plotting code:

* put method vocabulary changes in ``_methods.py``;
* put kwargs alias changes in ``_kwargs.py``;
* keep style decisions in ``_style.py``;
* keep artist creation in the plotter modules;
* avoid re-implementing normalization logic in wrappers;
* be careful with ``ax``, ``clear``, and ``show`` because composite plots and
  orchestration helpers can have narrower contracts than ``dataset.plot()``.
