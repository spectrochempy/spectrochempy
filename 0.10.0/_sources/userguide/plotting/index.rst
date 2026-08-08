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
