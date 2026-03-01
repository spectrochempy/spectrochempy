.. _contributing_plotting:

Plotting Architecture and Matplotlib Integration
================================================

Plotting in SpectroChemPy is built on top of Matplotlib but deliberately avoids
direct, uncontrolled manipulation of ``matplotlib.rcParams`` and pyplot state.

Instead, SpectroChemPy exposes a *typed, observable, and reversible* plotting
configuration layer based on ``traitlets``. This allows:

- reproducible plotting behavior
- safe application of Matplotlib style sheets
- controlled synchronization with ``rcParams``
- consistent behavior across scripts, notebooks, and GUIs

This document explains how plotting preferences are handled internally and how
contributors should extend or modify plotting behavior.

.. contents:: Contents
   :local:
   :depth: 2


High-Level Design
-----------------

The key components are:

- ``PlotPreferences`` (``spectrochempy.application._preferences.plot_preferences``)
- ``PreferencesSet`` (application-level preference management)
- Matplotlib ``rcParams`` (derived state, never authoritative)

**Important principle**

``rcParams`` are **never the source of truth**.
They are always a *projection* of the current ``PlotPreferences`` state.

Plotting preferences flow in one direction only:

::

    PlotPreferences (traitlets)
            ↓
      matplotlib.rcParams


Step 1: PlotPreferences as the Source of Truth
----------------------------------------------

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


Step 2: Trait Observation and rcParams Synchronization
------------------------------------------------------

Synchronization with Matplotlib happens **only** in one place:

.. code-block:: python

    @observe(All)
    def _anytrait_changed(self, change):
        ...

This observer:

1. detects changes on configurable traits
2. maps the trait name back to an rcParams key
3. updates ``matplotlib.rcParams`` accordingly

**Contributors must never write to ``rcParams`` directly**, except in very
specific legacy cases (e.g. LaTeX font handling).

Why this matters:

- ensures reversibility
- allows preferences reset
- avoids hidden global state changes


Step 3: Style Sheets Are Parsed, Not Applied
--------------------------------------------

Matplotlib style sheets (``.mplstyle`` files) are **not** applied using
``plt.style.use()``.

Instead, SpectroChemPy:

1. reads the style file line by line
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in ``_coerce_for_trait()``:

- adapts parsed values to the *target trait*
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

is kept as the string ``"None"`` for compatibility with Matplotlib.


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

Contributors adding new logical styles must handle them **before**
filesystem access.


Step 7: Adding or Modifying Plot Preferences
--------------------------------------------

When adding a new plotting option:

1. Add a trait to ``PlotPreferences``
2. Ensure its name maps correctly to an rcParams key (if applicable)
3. Choose the correct trait type:
   - ``Float`` / ``Integer`` for numeric values
   - ``Unicode`` for Matplotlib string semantics
   - ``Enum`` for constrained choices
   - ``TraitUnion`` for mixed types
4. Let traitlets perform validation
5. Do **not** write to ``rcParams`` directly

If the option does **not** map to Matplotlib:

- document it clearly as a SpectroChemPy-only option
- consume it in plotter code, not in ``PlotPreferences`` observers


Step 8: Testing and Debugging
-----------------------------

Recommended tests:

- applying the ``scpy`` style
- resetting preferences
- round-tripping style values
- ensuring ``rcParams`` are restored correctly

Common failure modes:

- passing real ``None`` to Enum traits
- converting strings too aggressively
- bypassing traitlets validation
- writing to ``rcParams`` outside observers


Summary for Contributors
------------------------

- ``PlotPreferences`` is authoritative
- ``rcParams`` are derived state
- styles are parsed, not blindly applied
- traitlets handle validation and observation
- all coercion is explicit and centralized

This architecture is intentionally verbose to protect users from
subtle, global plotting side effects.
