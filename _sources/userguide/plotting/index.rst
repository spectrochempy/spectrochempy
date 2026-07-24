Plotting
========

SpectroChemPy provides a high-level plotting interface built on top of
Matplotlib, tailored for scientific spectroscopy workflows.

Most scientific plots in SpectroChemPy require only **one line of code**.

The goal is simple:

   **Produce clear, publication-quality scientific plots with minimal code,
   while remaining fully compatible with Matplotlib.**

This approach is built on three core principles.


Sensible Defaults
-----------------

A good plot should not require configuration.

Calling::

    dataset.plot()

immediately produces properly labeled axes (with units), a readable layout,
consistent scientific styling, and appropriate color mapping for 2D data.

Plot configuration is automatically derived from dataset metadata
(titles, coordinates, units), so in most cases no additional code is required.


Progressive Customization
-------------------------

When adjustments are needed, they remain simple.

- Modify a single plot with keyword arguments.
- Change visual style without rewriting plotting code.
- Adjust global defaults via ``scp.preferences``.

Users can start with defaults and progressively gain control,
without modifying Matplotlib’s global configuration.


Full Matplotlib Compatibility
-----------------------------

SpectroChemPy does not replace Matplotlib — it builds on it.

Each plotting function returns a Matplotlib ``Axes`` object,
so advanced users retain full control over figure customization
and integration into complex layouts.

Current Plotting Contract
-------------------------

For day-to-day use, the plotting contract is:

- ``dataset.plot(method=...)`` selects the plotting geometry.
- Explicit helpers such as ``plot_pen()``, ``plot_scatter()``, ``plot_bar()``,
  ``plot_lines()``, ``plot_contour()``, ``plot_contourf()``, and
  ``plot_image()`` make that intent explicit.
- Common keyword aliases such as ``lw``, ``ls``, ``ms``, ``mew``, ``c``, and
  ``colormap`` are normalized internally to their canonical Matplotlib names.
- Style interpretation depends on the plotting geometry: the same ``cmap`` or
  ``marker`` input can mean different things for lines, scatter plots, contour
  plots, and image-like plots.
- ``ax``, ``clear``, and ``show`` control figure lifecycle for both ordinary
  dataset plots and   composite plots (``plot_score``, ``plot_scree``,
  ``plot_compare``, ``plot_merit``, ``plot_baseline``, ``plot_parity``).
  Here ``show`` means "perform SpectroChemPy's explicit display step after
  plotting", not "guarantee figure visibility". In notebook environments,
  figures can still render inline without that explicit call.
- ``plot_multiple()`` overlays several datasets on one axes, while
  ``multiplot()`` creates a grid of axes.


Designed for Scientific Workflows
---------------------------------

SpectroChemPy plotting is aware of:

- Physical units
- Dataset dimensionality
- Spectroscopic conventions
- Analysis and decomposition outputs

This reduces boilerplate code and keeps the focus on scientific interpretation.

In short, SpectroChemPy plotting should feel like Matplotlib —
but smarter, cleaner, and more efficient for spectroscopy.


Structure of This Section
--------------------------

The following pages introduce plotting progressively:

.. toctree::
   :maxdepth: 2

   overview
   customization
   preferences
   plot_types
   styles
   advanced
