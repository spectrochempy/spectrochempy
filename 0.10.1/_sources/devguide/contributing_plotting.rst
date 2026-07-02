.. _contributing_plotting:

Plotting Contributions
======================

This page summarizes the contributor-facing rules for working on plotting code.
It is intentionally practical. Deep implementation history and internal
architecture notes do not belong in the public developer guide.

Scope
-----

Use this guidance when you touch:

* plotting functions in ``src/spectrochempy/plotting/``
* ``NDDataset.plot()`` and related dispatch code
* plotting preferences and style handling
* tests that depend on Matplotlib state

Core Rules
----------

When contributing to plotting code:

* prefer lazy Matplotlib usage
* avoid global ``matplotlib.rcParams`` mutation
* apply preferences locally for each plot
* keep plotting behavior reproducible across scripts and notebooks
* preserve import-time performance for non-plotting workflows

In practice, this means:

* call the project's lazy Matplotlib setup before using heavy plotting APIs
* use local plotting contexts such as ``matplotlib.rc_context()``
* avoid importing ``matplotlib.pyplot`` at module import time unless there is a
  strong reason
* avoid hidden global state changes in helpers or tests

Lazy Matplotlib
---------------

SpectroChemPy tries to keep Matplotlib overhead low until plotting is actually
used.

Contributor rule:

* do not force eager Matplotlib initialization from non-plotting code paths

Typical pattern:

.. code-block:: python

    from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config

    def some_plot(...):
        lazy_ensure_mpl_config()
        import matplotlib as mpl
        with mpl.rc_context():
            ...

The exact internals may evolve, but the contributor expectation is stable:
heavy plotting setup should happen at plot time, not import time.

Preferences and Styles
----------------------

Plotting options should generally flow through the project's preference layer,
not through ad-hoc direct writes to Matplotlib globals.

Contributor rule:

* if an option is a stable user-facing plotting preference, define or reuse it
  in the preference system rather than writing directly to ``rcParams``

Styles should be treated as inputs to the SpectroChemPy plotting layer, not as
commands to mutate Matplotlib globally for the whole session.

What to Avoid
-------------

Avoid these patterns unless there is a very specific reason:

* module-level ``import matplotlib.pyplot as plt`` in generic code paths
* direct global writes such as ``mpl.rcParams[...] = ...``
* helper functions that silently modify plotting state outside the current plot
* tests that leave figures, backends, or rcParams mutated for later tests

Testing Guidance
----------------

Prefer focused tests around the change you made.

Useful kinds of plotting tests include:

* initialization happens only when plotting is used
* preferences are applied locally to the plot being created
* repeated plots do not leak state between calls
* import-only workflows do not pay unnecessary Matplotlib cost

If a test changes plotting state, make sure the state is restored through the
existing test fixtures and isolation helpers.

When to Go Deeper
-----------------

If a change requires redesigning plotting preferences, backend boundaries, or
global plotting semantics, that is no longer a small contributor-facing change.
At that point, prefer a maintainer discussion or an architectural note rather
than expanding this guide with implementation-specific detail.

Checklist
---------

Before opening a plotting PR, quickly verify:

* the change does not introduce unnecessary eager Matplotlib imports
* no global rcParams mutation was added by accident
* user-facing options are documented in the right place
* tests cover the changed plotting behavior
