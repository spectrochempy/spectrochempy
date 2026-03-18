.. _contributing_plotting:

Plotting Architecture and Lazy Matplotlib Integration
==================================================

Plotting in SpectroChemPy is built on top of Matplotlib but deliberately avoids
direct, uncontrolled manipulation of ``matplotlib.rcParams`` and pyplot state.
A key design is **lazy matplotlib initialization** - matplotlib is only fully
loaded when plotting is actually used, reducing import-time overhead for
non-plotting workflows.

Instead, SpectroChemPy exposes a *typed, observable, and reversible* plotting
configuration layer based on ``traitlets``. This allows:

- **Reduced import overhead** for non-plotting workflows
- **Lazy matplotlib loading** - minimal overhead until first plot
- reproducible plotting behavior
- safe application of Matplotlib style sheets
- local rcParams management in plotting contexts
- consistent behavior across scripts, notebooks, and GUIs

This document explains how plotting preferences are handled internally and how
contributors should extend or modify plotting behavior.

.. contents:: Contents
   :local:
   :depth: 2


High-Level Architecture
----------------------

The key components are:

- ``PlotPreferences`` (``spectrochempy.application._preferences.plot_preferences``)
- ``PreferencesSet`` (application-level preference management)
- **Lazy Initialization System** (``spectrochempy.core.plotters.plot_setup``)
- Matplotlib ``rcParams`` (applied locally in plot functions, not globally authoritative)

**Important principle**

``rcParams`` are **never** the source of truth.
They are set locally in plotting contexts, not globally synchronized.

**Lazy initialization principle**

Matplotlib is **minimized** during import (not fully eliminated).
The initialization system is triggered on the first actual plotting operation.

Plotting preferences flow in one direction only:

::

    PlotPreferences (traitlets)
            ↓
    Plot Function (local rc_context)
            ↓
    Temporary rcParams (plot-specific)

The matplotlib import is lazy in the sense that heavy imports (pyplot, backends)
are deferred. However, some plotting modules have unavoidable matplotlib imports
at module level for reading defaults.

Actual Architecture (Simplified)
-------------------------------

The current implementation uses a simplified architecture:

1. ``PlotPreferences`` defines all configuration as traitlets
2. Plot functions read preferences and apply them via ``matplotlib.rc_context``
3. Changes are local to each plot call, not globally enforced
4. No persistent global rcParams modification occurs

This approach:

- Avoids global side effects
- Works regardless of import order
- Doesn't require complex state management
- Is thread-safe by design (each plot is independent)

Example flow in a plotting function:

.. code-block:: python

    def plot(self, ...):
        # Ensure matplotlib is available
        lazy_ensure_mpl_config()

        # Get preferences
        prefs = preferences.plot

        # Apply locally for this plot
        import matplotlib as mpl
        with mpl.rc_context():
            prefs.apply_to_rcparams()
            # ... rest of plotting logic


Lazy Initialization System
--------------------------

SpectroChemPy implements lazy matplotlib initialization to reduce import overhead.

Core Components
^^^^^^^^^^^^^^

The lazy system is implemented in ``spectrochempy.core.plotters.plot_setup``:

.. code-block:: python

    # Simple flag to track initialization state
    _MPL_READY: bool = False

    def lazy_ensure_mpl_config():
        """Ensure matplotlib is initialized. Idempotent and fast on subsequent calls."""
        global _MPL_READY
        if _MPL_READY:
            return
        _MPL_READY = True

Implementation Details
^^^^^^^^^^^^^^^^^^^^^

- **State tracking**: A simple boolean flag (``_MPL_READY``), not a complex state machine
- **No threading lock**: The current implementation is single-threaded; the lock was designed but not implemented
- **Idempotent**: Safe to call multiple times; returns immediately if already initialized
- **Minimal**: Only sets the ready flag; actual matplotlib setup happens on first use

Trigger Points
^^^^^^^^^^^^^

Matplotlib initialization is triggered from multiple entry points:

.. code-block:: python

    # lazy_ensure_mpl_config() is called from:
    # - plotting/multiplot.py (multi-panel plots)
    # - plotting/plot2d.py (2D plots)
    # - plotting/plot1d.py (1D plots)
    # - plotting/backends/matplotlib_backend.py (backend dispatcher)
    # - core/dataset/nddataset.py (dataset.plot() method)

    def plot(self, *args, **kwargs):
        lazy_ensure_mpl_config()  # Triggers initialization
        # ... rest of plotting logic

This is not an exhaustive list. Any plotting function may call this function
to ensure matplotlib is available.

Benefits
^^^^^^^^

- **Import performance**: Reduced overhead for non-plotting workflows
- **Memory efficiency**: Full matplotlib loaded only when needed
- **Simplicity**: Boolean flag is easy to understand and maintain
- **Local application**: No global state modification

Import Behavior Clarification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The lazy system **reduces** but does **not fully eliminate** matplotlib imports at module load time:

- Main ``spectrochempy/__init__.py`` is clean - no matplotlib imports
- Some plotting modules (``plotting/_style.py``, ``plotting/_colorbar_utils.py``)
  have module-level imports for reading defaults
- These imports are acceptable because they don't load heavy components (pyplot, backends)
- Lazy loading ensures heavy imports only happen on first plot

Example of acceptable module-level import:

.. code-block:: python

    # In plotting/_style.py
    import matplotlib as mpl  # Lightweight import for defaults only
    _MPL_DEFAULT_IMAGE_CMAP = mpl.rcParamsDefault["image.cmap"]


Step 1: PlotPreferences as Source of Truth
------------------------------------------

All Matplotlib-related options are defined as traitlets in ``PlotPreferences``.

Example:

.. code-block:: python

    axes_linewidth = Float(0.8).tag(config=True)
    lines_marker = Enum(list(Line2D.markers.keys()), default_value="None")

Key properties:

- each trait corresponds to **one rcParams key**
- trait names replace ``.`` with ``_`` (e.g. ``axes.facecolor`` → ``axes_facecolor``)
- defaults are *SpectroChemPy defaults*, not necessarily Matplotlib defaults

Mapping back to Matplotlib keys is handled automatically:

.. code-block:: python

    axes_facecolor  →  "axes.facecolor"


Step 2: Local rcParams Application
----------------------------------

Synchronization with Matplotlib happens **locally** within plot functions, not globally.

**Important change**: The global observer ``@observe(All)`` that previously synchronized
PlotPreferences to rcParams is **disabled**. Instead, plotting functions apply
preferences locally using ``rc_context``.

.. code-block:: python

    @observe(All)  # DISABLED - do not use
    def _anytrait_changed(self, change):
        # No longer automatically applies to global rcParams
        pass

This design:

- Prevents global side effects
- Avoids race conditions with concurrent plots
- Allows preferences reset without affecting matplotlib globally
- Works regardless of when matplotlib was initialized

Contributors should **never write to ``rcParams`` directly**. Instead:

1. Read preferences from ``PlotPreferences``
2. Apply them locally using ``rc_context`` in plotting functions


Step 3: Style Sheets Are Parsed, Not Applied
---------------------------------------------

Matplotlib style sheets (``.mplstyle`` files) are **not** applied using
``plt.style.use()``.

Instead, SpectroChemPy:

1. reads style file line by line
2. parses each ``key: value`` pair
3. converts values to proper Python types
4. assigns them to traitlets

This happens in ``PlotPreferences._apply_style()``.

Example:

.. code-block:: python

    lines.linewidth : 0.75
    agg.path.chunksize : 20000.0

are converted and validated before reaching rcParams.

This design avoids:

- silent type coercion by Matplotlib
- global side effects
- invalid style values slipping through
- premature matplotlib loading


Step 4: Two-Stage Value Coercion
--------------------------------

Style values go through **two explicit coercion stages**.

Stage 1: Semantic parsing
^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in ``_coerce_style_value()``:

- converts strings to booleans, numbers, tuples
- handles ``"None"`` and ``"null"`` safely
- remains *trait-aware*

Example:

.. code-block:: python

    "20000.0" → 20000.0
    "true"    → True
    "5.5, 3.5" → (5.5, 3.5)


Stage 2: Trait normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in ``_coerce_for_trait()``:

- adapts parsed values to *target trait*
- preserves strings for ``Unicode`` traits
- promotes floats to ints where appropriate
- delegates final validation to traitlets

This separation is intentional and prevents fragile,
hard-to-debug parsing logic.


Step 5: Handling Special Matplotlib Semantics
---------------------------------------------

Matplotlib uses several string-based sentinels:

- ``"None"``
- ``"auto"``
- ``"inherit"``
- marker names like ``"None"`` or ``""``

SpectroChemPy preserves these semantics by:

- keeping strings for ``Unicode`` and ``Enum`` traits
- avoiding conversion to real ``None`` unless explicitly allowed

Example:

.. code-block:: python

    legend.framealpha : None

is kept as string ``"None"`` for compatibility with Matplotlib.


Step 6: Logical vs File-Based Styles
-------------------------------------

Some Matplotlib styles are *logical*, not file-based.

Example:

- ``default``

These are handled explicitly:

.. code-block:: python

    if _style == "default":
        # Reset to matplotlib defaults locally
        return

Attempting to load ``default.mplstyle`` from disk would fail.

Contributors adding new logical styles must handle them **before**
filesystem access.


Step 7: Adding or Modifying Plot Preferences
--------------------------------------------

When adding a new plotting option:

1. Add a trait to ``PlotPreferences``
2. Ensure its name maps correctly to an rcParams key (if applicable)
3. Choose correct trait type:
   - ``Float`` / ``Integer`` for numeric values
   - ``Unicode`` for Matplotlib string semantics
   - ``Enum`` for constrained choices
   - ``TraitUnion`` for mixed types
4. Let traitlets perform validation
5. Do **not** write to ``rcParams`` directly
6. Use ``rc_context`` in plotting functions to apply preferences locally

If option does **not** map to Matplotlib:

- document it clearly as a SpectroChemPy-only option
- consume it in plotter code, not in ``PlotPreferences`` observers
- ensure it doesn't trigger unnecessary matplotlib loading


Step 8: Testing and Debugging
-----------------------------

Recommended tests for plotting code:

- matplotlib should not be loaded during import (if testing this)
- plot functions trigger initialization correctly
- preferences work correctly when applied locally
- style application works correctly
- multiple plots don't interfere with each other

Common failure modes:

- importing matplotlib at module level unnecessarily
- bypassing the lazy initialization system
- writing to ``rcParams`` globally instead of using rc_context
- triggering matplotlib initialization in non-plotting code


Performance Guidelines for Contributors
-------------------------------------

When contributing to plotting code:

**DO:**
- Use the lazy initialization system
- Check matplotlib state before accessing heavy matplotlib APIs
- Apply preferences locally using rc_context
- Test import performance impact

**DON'T:**
- Import matplotlib at module level if avoidable
- Access ``plt`` or heavy matplotlib modules without lazy guards
- Call heavy matplotlib functions during import
- Modify global rcParams

Example of safe matplotlib usage:

.. code-block:: python

    def some_plotting_function():
        from spectrochempy.core.plotters.plot_setup import _is_mpl_initialized

        if not _is_mpl_initialized():
            # Trigger lazy initialization
            lazy_ensure_mpl_config()

        # Now safe to use matplotlib locally
        import matplotlib as mpl
        with mpl.rc_context():
            # Apply preferences and plot
            ...


Summary for Contributors
------------------------

- **Lazy first**: matplotlib is initialized when actually needed
- **PlotPreferences** defines configuration via traitlets
- **rcParams** are applied locally, not globally synchronized
- styles are parsed, not blindly applied
- traitlets handle validation and observation
- all coercion is explicit and centralized
- no global state modification means thread safety by default
- performance impact should be considered

This architecture is intentionally simpler than originally designed to protect users from
subtle, global plotting side effects while providing a maintainable system for
lazy matplotlib initialization.
