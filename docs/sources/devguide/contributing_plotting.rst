.. _contributing_plotting:

Plotting Architecture and Lazy Matplotlib Integration
==================================================

Plotting in SpectroChemPy is built on top of Matplotlib but deliberately avoids
direct, uncontrolled manipulation of ``matplotlib.rcParams`` and pyplot state.
A key innovation is **lazy matplotlib initialization** - matplotlib is only loaded
when plotting is actually used, providing dramatic performance improvements for
non-plotting workflows.

Instead, SpectroChemPy exposes a *typed, observable, and reversible* plotting
configuration layer based on ``traitlets``. This allows:

- **46% faster import times** for non-plotting workflows
- **Lazy matplotlib loading** - zero overhead until first plot
- reproducible plotting behavior
- safe application of Matplotlib style sheets
- controlled synchronization with ``rcParams``
- consistent behavior across scripts, notebooks, and GUIs

This document explains how plotting preferences are handled internally and how
contributors should extend or modify plotting behavior.

.. contents:: Contents
   :local:
   :depth: 2


High-Level Architecture
---------------------

The key components are:

- ``PlotPreferences`` (``spectrochempy.application._preferences.plot_preferences``)
- ``PreferencesSet`` (application-level preference management)
- **Lazy Initialization System** (``spectrochempy.core.plotters.plot_setup``)
- Matplotlib ``rcParams`` (derived state, never authoritative)

**Important principle**

``rcParams`` are **never** source of truth.
They are always a *projection* of current ``PlotPreferences`` state.

**Lazy initialization principle**

Matplotlib is **never** loaded during import.
It's only initialized on the first actual plotting operation.

Plotting preferences flow in one direction only:

::

    PlotPreferences (traitlets)
            ↓
      matplotlib.rcParams

But matplotlib itself follows a lazy pattern:

::

    Import Time:  No matplotlib loaded
            ↓
    First Plot:  Lazy initialization → matplotlib loaded
            ↓
    Subsequent Plots: matplotlib ready


Lazy Initialization System
--------------------------

SpectroChemPy now implements comprehensive lazy matplotlib initialization to eliminate
import-time overhead while preserving full functionality.

Core Components
^^^^^^^^^^^^^^^^^

The lazy system is implemented in ``spectrochempy.core.plotters.plot_setup``:

.. code-block:: python

    # State tracking enum
    class MPLInitState(Enum):
        NOT_INITIALIZED = "not_initialized"
        INITIALIZING = "initializing"
        INITIALIZED = "initialized"
        FAILED = "failed"

    # Global state and thread safety
    _MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED
    _MPL_INIT_LOCK = threading.RLock()

    # Main lazy initialization function
    def lazy_ensure_mpl_config():
        """Comprehensive matplotlib setup with lazy initialization."""

Trigger Points
^^^^^^^^^^^^^^

Matplotlib is only initialized in one place:

.. code-block:: python

    # In ndplot.py - the single trigger point
    def plot(self, *args, **kwargs):
        lazy_ensure_mpl_config()  # Only here!
        # ... rest of plotting logic

Benefits
^^^^^^^^^

- **Import performance**: 46% faster (232ms → 126ms)
- **Memory efficiency**: matplotlib only loaded when needed
- **Thread safety**: Proper locking prevents race conditions
- **Preference deferral**: Changes before init are queued and applied later


Step 1: PlotPreferences as Source of Truth
--------------------------------------------

All Matplotlib-related options are mirrored as traitlets in
``PlotPreferences``.

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


Step 2: Lazy Trait Observation and rcParams Synchronization
----------------------------------------------------------

Synchronization with Matplotlib happens **only** in one place, but now with
lazy awareness:

.. code-block:: python

    @observe(All)
    def _anytrait_changed(self, change):
        # Queue changes if matplotlib not yet initialized
        if not _is_mpl_initialized():
            _defer_preference_change(change)
            return

        # Apply immediately if matplotlib is ready
        _apply_preference_change(change)

This observer has two modes:

1. **Lazy mode** (before matplotlib init): Changes are queued
2. **Immediate mode** (after matplotlib init): Changes applied to rcParams

**Contributors must never write to ``rcParams`` directly**, except in very
specific legacy cases (e.g. LaTeX font handling).

Why this matters with lazy loading:

- ensures preferences are preserved even before matplotlib loads
- allows preferences reset regardless of initialization state
- maintains reversibility across lazy initialization boundary


Step 3: Style Sheets Are Parsed, Not Applied
--------------------------------------------

Matplotlib style sheets (``.mplstyle`` files) are **not** applied using
``plt.style.use()``.

Instead, SpectroChemPy:

1. reads style file line by line
2. parses each ``key: value`` pair
3. converts values to proper Python types
4. assigns them to traitlets

This happens in ``PlotPreferences._apply_style()`` and works seamlessly with
lazy initialization - styles are parsed immediately but only applied to matplotlib
when it's loaded.

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
^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
------------------------------------

Some Matplotlib styles are *logical*, not file-based.

Example:

- ``default``

These are handled explicitly:

.. code-block:: python

    if _style == "default":
        plt.rcdefaults()
        return

Attempting to load ``default.mplstyle`` from disk would fail.

With lazy initialization, logical styles are handled during the lazy setup process,
ensuring matplotlib is available before ``plt.rcdefaults()`` is called.

Contributors adding new logical styles must handle them **before**
filesystem access.


Step 7: Lazy Preference Deferral
--------------------------------

A key innovation with lazy initialization is **preference deferral**:

Before matplotlib is initialized:

.. code-block:: python

    def _defer_preference_change(change):
        """Queue preference changes until matplotlib is ready."""
        _PENDING_PREFERENCE_CHANGES.append(change)

After matplotlib is initialized:

.. code-block:: python

    def _apply_deferred_preferences():
        """Apply all queued preference changes."""
        for change in _PENDING_PREFERENCE_CHANGES:
            _apply_preference_change(change)
        _PENDING_PREFERENCE_CHANGES.clear()

This ensures:

- User preferences are never lost, even if set before matplotlib loads
- Order of preference changes is preserved
- Seamless experience regardless of initialization timing


Step 8: Adding or Modifying Plot Preferences
----------------------------------------------

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
6. **Consider lazy compatibility**: Your changes should work whether matplotlib is loaded or not

If option does **not** map to Matplotlib:

- document it clearly as a SpectroChemPy-only option
- consume it in plotter code, not in ``PlotPreferences`` observers
- ensure it doesn't trigger premature matplotlib loading


Step 9: Thread Safety and Concurrency
-------------------------------------

Lazy initialization must be thread-safe:

.. code-block:: python

    def lazy_ensure_mpl_config():
        with _MPL_INIT_LOCK:
            if _MPL_INIT_STATE == MPLInitState.INITIALIZED:
                return
            elif _MPL_INIT_STATE == MPLInitState.NOT_INITIALIZED:
                _perform_initialization()

Contributors must:

- Always use the lock when accessing matplotlib state
- Respect the initialization state machine
- Never bypass the lazy system


Step 10: Testing and Debugging
-------------------------------

Recommended tests for lazy initialization:

- import performance (matplotlib should not be loaded)
- first plot triggers initialization correctly
- preferences work before and after initialization
- concurrent initialization is thread-safe
- style application works in lazy context

Common failure modes:

- accidentally importing matplotlib during module load
- bypassing the lazy initialization system
- forgetting to thread-lock matplotlib access
- writing to ``rcParams`` outside the deferred system
- triggering matplotlib initialization in non-plotting code


Performance Guidelines for Contributors
---------------------------------------

When contributing to plotting code:

**DO:**
- Use the lazy initialization system
- Check matplotlib state before accessing matplotlib APIs
- Queue operations that require matplotlib if it's not loaded
- Test import performance impact

**DON'T:**
- Import matplotlib at module level
- Access ``plt`` or ``matplotlib`` without lazy guards
- Call matplotlib functions during import
- Assume matplotlib is always available

Example of safe matplotlib usage:

.. code-block:: python

    def some_plotting_function():
        from spectrochempy.core.plotters.plot_setup import _is_mpl_initialized

        if not _is_mpl_initialized():
            # Queue operation or trigger lazy init
            lazy_ensure_mpl_config()

        # Now safe to use matplotlib
        import matplotlib.pyplot as plt
        plt.plot([...])


Summary for Contributors
------------------------

- **Lazy first**: matplotlib only loads when actually needed
- **PlotPreferences** is authoritative
- **rcParams** are derived state
- styles are parsed, not blindly applied
- traitlets handle validation and observation
- all coercion is explicit and centralized
- thread safety is mandatory
- performance impact must be considered

This architecture is intentionally verbose to protect users from
subtle, global plotting side effects while providing dramatic
performance improvements through lazy initialization.

The lazy initialization system delivers **46% faster import times**
while maintaining 100% backward compatibility and full plotting functionality.
