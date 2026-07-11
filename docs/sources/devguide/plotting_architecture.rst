.. _plotting-architecture:

Plotting Architecture
=====================

This note describes the plotting architecture that exists today after the
recent cleanup work.  Its purpose is to explain the current responsibility
boundaries so that future plotting changes remain focused and consistent.

It is not a feature guide and it is not an RFC.

Motivation
----------

The plotting cleanup was driven by accumulated maintenance problems:

* public plotting regressions caused by duplicated dispatch and wrapper logic;
* inconsistent handling of method aliases and plotting kwargs;
* normalization rules spread across several modules;
* wrappers partially re-implementing backend behavior.

The resulting architecture intentionally separates:

* plotting vocabulary;
* kwargs normalization;
* style interpretation;
* artist creation;
* specialized orchestration.

Current Architecture
--------------------

The main public plotting surface includes:

* ``NDDataset.plot()`` for dimension-aware default dispatch;
* explicit helpers such as ``plot_pen()``, ``plot_scatter()``,
  ``plot_bar()``, ``plot_lines()``, ``plot_contour()``,
  ``plot_contourf()``, ``plot_image()``, ``plot_surface()``, and
  ``plot_waterfall()``;
* orchestration helpers such as ``plot_multiple()`` and ``multiplot()``;
* composite plotters and plugin plotters, which build on the shared plotting
  layer but may own additional specialized behavior.

The normal dataset plotting path is:

.. code-block:: text

    NDDataset.plot()
        -> plotting.dispatcher.plot_dataset()
        -> plotting.backends.matplotlib_backend.plot_dataset_impl()
        -> plotting._methods
        -> plotting._kwargs
        -> plotting._style
        -> plot1d.py / plot2d.py / plot3d.py
        -> Matplotlib artists

Read that flow as a sequence of responsibilities, not as a promise that every
function literally calls each helper in exactly that textual order for every
plot family.

A simplified architecture sketch is:

.. code-block:: text

    User API
        |
        v
    Dispatcher
        |
        v
    Backend
        |
        +-- _methods.py   (semantic normalization)
        +-- _kwargs.py    (parameter normalization)
        +-- _style.py     (style interpretation)
                  |
                  v
            plot1d / plot2d / plot3d
                  |
                  v
          Matplotlib artists

Responsibility Boundaries
-------------------------

``plotting._methods``
~~~~~~~~~~~~~~~~~~~~~

This module owns plotting vocabulary normalization:

* canonical method names;
* compatibility aliases such as ``lines <-> pen`` where appropriate;
* dimensional aliases such as ``stack -> lines`` and ``map -> contour``;
* validation of supported method names for the relevant plotting family.

This module does not decide styling or create artists.

``plotting._kwargs``
~~~~~~~~~~~~~~~~~~~~

This module owns plotting kwargs normalization:

* alias renaming such as ``lw -> linewidth`` and ``ls -> linestyle``;
* marker-related aliases such as ``ms -> markersize`` and
  ``mew -> markeredgewidth``;
* color aliases such as ``c -> color`` and ``colormap -> cmap``;
* production of canonical kwargs for downstream plotting code.

This module does not interpret what those values should mean visually.

``plotting._style``
~~~~~~~~~~~~~~~~~~~

This module owns style interpretation after normalization:

* geometry-dependent marker defaults;
* line defaults and palette decisions;
* colormap and alpha interpretation;
* conversion of normalized plotting inputs into concrete style choices.

This is the layer that answers questions such as "what marker should a scatter
plot use by default?" rather than "what is the canonical spelling of this
kwarg?".

``plot1d.py`` / ``plot2d.py`` / ``plot3d.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These plotter modules own rendering work:

* creation of Matplotlib artists;
* geometry-specific axis policy;
* application of normalized methods, kwargs, and interpreted style to concrete
  artists.

This is where rendering happens.  These modules should not duplicate
normalization logic that belongs in ``_methods.py`` or ``_kwargs.py``.

Dispatcher and Backend
~~~~~~~~~~~~~~~~~~~~~~

The dispatcher selects the active plotting backend.  The Matplotlib backend
owns the standard dataset plotting dispatch for that backend and the top-level
``show`` handling for that path.

This layer is responsible for sending an already-understood plotting request to
the correct rendering family.  It is not the place to redefine plotting
vocabulary or kwargs aliases.

Composite Plotters
~~~~~~~~~~~~~~~~~~

Domain-specific composite functions live in
``plotting/composite/`` and implement purpose-built visualisations:

* ``plot_score`` — PCA scores scatter (2D or 3D);
* ``plot_scree`` — explained variance bar + cumulative line;
* ``plot_merit`` — original vs reconstructed vs residual;
* ``plot_compare`` — generic two-dataset overlay with residual;
* ``plot_baseline`` — two-panel baseline correction view;
* ``plot_parity`` — predicted vs measured scatter.

These functions form a **middle layer** between scientific objects (``PCA``,
``PLSRegression``, etc.) and the dataset plotting stack below:

.. code-block:: text

    Scientific objects (PCA, PLSRegression, ...)
        │  thin wrapper with domain-specific defaults
        ▼
    Composite plotters (plotting/composite/)
        │  own figure, axes, rendering; shared lifecycle helpers
        ▼
    Dataset plotting (plot1d, plot2d, dispatcher, _style, …)
        │  generic layered pipeline
        ▼
    Matplotlib artists

Key conventions for composite plotters:

* Accept ``ax``, ``clear``, ``show`` and use the shared lifecycle helpers
  (``_setup_axes`` / ``_maybe_show`` from ``mplutils.py``).
* Return a single ``Axes`` object (exception: ``plot_baseline`` returns a
  tuple of two ``Axes`` for its two-panel layout).
* Own their own rendering — they do NOT go through the dispatcher or
  ``_render.py`` primitives (though ``plot_compare`` and ``plot_baseline``
  use ``render_lines`` where appropriate).
* Style is resolved locally with explicit per-function parameters, not
  through ``_style.py`` resolvers (except ``plot_baseline`` which uses
  ``resolve_stack_colors``).
* Lifecycle contract is identical to dataset plotting: ``ax=None`` creates
  a new figure, ``ax + clear=True`` clears before plotting, ``ax +
  clear=False`` appends artists.

These conventions are verified by structural tests
(``test_composite_lifecycle.py``, ``test_parity.py``).

Style resolution follows a consistent priority chain across all plot types:

.. code-block:: text

    Explicit per-function parameters
        ↓ (if None)
    Composite-level defaults / kwargs normalization
        ↓ (if None)
    Global preferences (prefs.*)
        ↓ (if not applicable)
    Matplotlib rcParams defaults

Existing composite plotters intentionally do not reuse ``_style.py``
resolvers unless the semantic contract genuinely matches.  For example,
``resolve_stack_colors`` is appropriate for ``plot_baseline`` (baseline
stacks are semantically equivalent to dataset stacks), but
``resolve_line_style`` would not be appropriate for ``plot_compare``
because composites need three semantically distinct styles (experimental /
calculated / residual) per category.

Specialized Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~

Other specialized entry points include:

* ``plot_multiple()`` overlays several datasets on one axes;
* ``multiplot()`` builds grids of axes and delegates panel rendering;
* plugin plotting entry points may adapt plugin-specific objects or workflows
  before delegating to the shared plotting layer.

These helpers are orchestration code.  They should reuse the shared
normalization and rendering layers rather than re-implement them.

Design Principles
-----------------

The cleanup adopted a small set of explicit design principles:

Normalize first
~~~~~~~~~~~~~~~

Method aliases and kwargs aliases should be normalized before downstream code
interprets or renders them.

Interpret later
~~~~~~~~~~~~~~~

Style decisions belong after normalization, when the plotting geometry and
canonical parameters are already known.

Render last
~~~~~~~~~~~

Artist creation should be the final step.  Rendering code should receive
already-normalized, already-interpreted inputs whenever possible.

Avoid duplicated normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a method alias or kwargs alias needs to change, contributors should update
the central helper module rather than patching several wrappers independently.

Keep semantics separate from rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting semantics, normalization, style interpretation, and artist creation
are related, but they are not the same responsibility.  The current structure
exists to keep those concerns legible.

Architecture Rules
------------------

The following rules are intended to help future contributors preserve the
current structure:

1. Public plotting methods must not duplicate method normalization.
   Use ``_methods.py``.
2. Public plotting methods must not duplicate kwargs alias handling.
   Use ``_kwargs.py``.
3. Style decisions belong in ``_style.py``.
4. Plotter modules create artists; they do not normalize public API aliases.
5. Specialized plotters such as ``multiplot()``, composite plotters, and
   plugin plotters may orchestrate figures, but they should reuse the common
   normalization layers.
6. New plotting methods should fit into the existing responsibility split
   rather than introducing new local normalization logic.

What Was Intentionally Not Centralized
--------------------------------------

Not every plotting responsibility should live in one helper layer.

The following remain intentionally local or specialized:

* subplot layout and panel orchestration in ``multiplot()``;
* overlay-specific behavior in ``plot_multiple()``;
* composite visualizations that combine several plot types or analysis results;
* plugin-specific plotting entry points and adapters;
* parts of the figure/axes lifecycle that are inherently tied to specialized
  orchestration.

Centralizing these behaviors prematurely would blur the distinction between
shared plotting semantics and higher-level composition logic.

Figure and Axes Lifecycle
-------------------------

Figure and axes ownership follows a consistent pattern across all plot types:

* The main dataset plotting path uses ``NDDataset._figure_setup()`` to decide
  whether to create a figure, reuse the current figure, or use an explicit
  ``ax``.
* Composite plotters use the shared ``_setup_axes`` / ``_maybe_show`` helpers
  (``mplutils.py``) which implement the same ``ax`` / ``clear`` / ``show``
  contract.
* ``ax``, ``clear``, and ``show`` are the universal lifecycle controls across
all plot types — dataset plots, composite plots, ``plot_multiple``, and
   ``plot_parity`` all follow the same conventions.
* The shared helpers keep lifecycle logic in one place and prevent the
  duplication that previously existed (five near-identical ``if ax is None``
  + ``clear`` + ``show`` patterns).

Practical Contributor Guidance
------------------------------

When changing plotting code:

* put method vocabulary changes in ``_methods.py``;
* put kwargs alias changes in ``_kwargs.py``;
* keep style interpretation in ``_style.py``;
* keep artist creation in the plotter modules;
* avoid re-implementing normalization in wrappers or composite helpers;
* treat ``plot_multiple()`` and ``multiplot()`` as orchestration layers, not
  as places to invent alternate plotting semantics.

Future Evolution
----------------

The current architecture is stable. Recent work (2026-07) has:

* unified the figure/axes lifecycle contract across ordinary and composite
  plots via ``_setup_axes`` / ``_maybe_show``;
* removed the previously dead ``use_plotly`` code path;
* added ``marker``/``s``/``alpha`` forwarding and kwargs normalization to
  key composite functions;
* aligned ``plot_parity`` and ``plot_multiple`` with the shared lifecycle
  contract.

Areas that may deserve future attention:

* possible convergence of plugin-side helper patterns (IRIS, NMR, Tensor)
  where duplication persists;
* future rendering abstractions only if they solve a demonstrated maintenance
  problem.

Any future evolution should preserve the current responsibility split unless a
clear replacement architecture is intentionally adopted.
