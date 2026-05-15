.. _plugin-architecture:

=====================
Writing a Plugin
=====================

This document explains how to write a plugin for SpectroChemPy.

.. contents:: :local:


How plugins work
================

Plugins are external Python packages that register readers, writers, and
processors with SpectroChemPy.  They are discovered automatically via
Python entry points (``spectrochempy.plugins`` group) at import time.

A plugin declares its contributions by implementing *declarative hooks*
--- simple methods that return data instead of mutating shared state::

    class MyPlugin:
        def register_readers(self) -> list[dict]:
            return [{"name": "myfmt", "func": read_myfmt, ...}]

The :class:`~spectrochempy.plugins.manager.PluginManager` collects these
contributions and registers them into a
:class:`~spectrochempy.plugins.registry.PluginRegistry`.

The registry is decomposed into specialised sub-registries, each
owning a single domain:

+----------------------+--------------------------------------------------+
| Sub-registry         | Responsibilities                                  |
+======================+==================================================+
| ``registry.io``      | Readers, writers, filetype associations           |
+----------------------+--------------------------------------------------+
| ``registry.processing`` | Processors, unit contexts, dtype handlers      |
+----------------------+--------------------------------------------------+
| ``registry.visualization`` | Visualizers (future use)                   |
+----------------------+--------------------------------------------------+
| ``registry.metadata``    | Plugin descriptors                           |
+----------------------+--------------------------------------------------+

Legacy code using top-level methods (``registry.register_reader(...)``)
continues to work via forwarding.  New code can target sub-registries
directly::

    registry.io.register_reader(...)
    registry.processing.register_processor(...)
    registry.metadata.register_plugin(...)


Minimal plugin example
======================

Here is the smallest possible plugin::

    # myplugin/src/myplugin/__init__.py

    from spectrochempy.api.plugins import SpectroChemPyPlugin


    class MyPlugin(SpectroChemPyPlugin):
        name = "myplugin"
        version = "0.1.0"

        def register_readers(self) -> list[dict]:
            return [
                {
                    "name": "myformat",
                    "func": read_myformat,
                    "description": "Read MyFormat files",
                    "extensions": [".myf", ".myformat"],
                },
            ]

And the corresponding ``pyproject.toml`` entry point::

    # myplugin/pyproject.toml

    [project.entry-points."spectrochempy.plugins"]
    myplugin = "myplugin:MyPlugin"

That is all.  Once installed, ``import spectrochempy`` will discover
``MyPlugin``, validate it, and register ``read_myformat`` as a reader.


Available hooks
===============

A plugin can implement any combination of the following methods:

``register_readers() -> list[dict]``
    Declare file readers.  Each dict must contain ``"name"`` and
    ``"func"``.  Optional keys: ``"description"`` (str),
    ``"extensions"`` (list[str]).

    ::

        def register_readers(self) -> list[dict]:
            return [
                {
                    "name": "myformat",
                    "func": read_myformat,
                    "description": "Read MyFormat files",
                    "extensions": [".myf"],
                },
            ]

``register_writers() -> list[dict]``
    Declare file writers.  Each dict must contain ``"name"`` and
    ``"func"``.  Optional key: ``"description"`` (str).

    ::

        def register_writers(self) -> list[dict]:
            return [
                {
                    "name": "myformat",
                    "func": write_myformat,
                    "description": "Write MyFormat files",
                },
            ]

``register_processors() -> list[dict]``
    Declare data processors.  Each dict must contain ``"name"`` and
    ``"func"``.  Optional key: ``"description"`` (str).

    ::

        def register_processors(self) -> list[dict]:
            return [
                {
                    "name": "smooth",
                    "func": smooth_data,
                    "description": "Smooth a signal",
                },
            ]


Returning an empty list (or ``None``) from a hook is treated as "no
contribution" and is silently ignored.


Standard capability names
=========================

You can attach a ``capabilities`` class attribute to advertise what
your plugin provides (purely informational)::

    from spectrochempy.api.plugins import PluginCapability

    class MyPlugin(SpectroChemPyPlugin):
        capabilities = [PluginCapability.READER, PluginCapability.WRITER]

Available capability values:

+------------------------+-----------+
| Enum member            | Value     |
+========================+===========+
| ``PluginCapability.READER``    | ``"reader"``    |
+------------------------+-----------+
| ``PluginCapability.WRITER``    | ``"writer"``    |
+------------------------+-----------+
| ``PluginCapability.PROCESSOR`` | ``"processor"`` |
+------------------------+-----------+
| ``PluginCapability.VISUALIZER``| ``"visualizer"``|
+------------------------+-----------+


Plugin metadata
===============

SpectroChemPy validates every plugin at registration time.  The
following class attributes are required:

``name`` (str)
    Unique plugin identifier.  Used as the entry point name.

``version`` (str)
    Plugin version (semver recommended).

The following are optional but recommended:

``description`` (str)
    Human-readable description of what the plugin does.

``spectrochempy_min_version`` (str)
    Minimum SpectroChemPy version required (e.g. ``"1.0"``).

``PLUGIN_API_VERSION`` (str)
    Plugin API version.  Defaults to ``"1.0"``.  Only the major
    version is checked for compatibility.

If you need to provide dynamic metadata, override ``plugin_info()``::

    def plugin_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "plugin_api_version": self.PLUGIN_API_VERSION,
            "spectrochempy_min_version": self.spectrochempy_min_version,
            "description": self.description,
        }


Full example: topspin plugin
============================

The reference external plugin is ``spectrochempy-topspin``::

    # spectrochempy-topspin/src/spectrochempy_topspin/__init__.py

    from spectrochempy.api.plugins import SpectroChemPyPlugin

    from .read_topspin import read_topspin


    class TopSpinPlugin(SpectroChemPyPlugin):
        name = "topspin"
        version = "0.1.0"

        def register_readers(self) -> list[dict]:
            return [
                {
                    "name": "topspin",
                    "func": read_topspin,
                    "description": "Bruker TOPSPIN fid, series, or processed data",
                    "extensions": [".fid", ".ser", "1r", "1i"],
                },
            ]

Entry point declaration in ``pyproject.toml``::

    [project.entry-points."spectrochempy.plugins"]
    topspin = "spectrochempy_topspin:TopSpinPlugin"


Import guidance
===============

Import from the stable public API namespace:

+------------------------------------------+------------------------------------------+
| ✅ Recommended                           | ❌ Avoid                              |
+==========================================+==========================================+
| ``from spectrochempy.api.plugins import`` | ``from spectrochempy.plugins import``    |
+------------------------------------------+------------------------------------------+

The public API (``spectrochempy.api``) is stable across releases.
Internal modules (``spectrochempy.plugins``) may change without notice.


Plugin lifecycle
=================

Every plugin managed by ``PluginManager`` passes through explicit
lifecycle states:

+---------------------+----------------------------------------------------+
| State               | Meaning                                            |
+=====================+====================================================+
| ``DISCOVERED``      | Entry point found via ``importlib.metadata``       |
+---------------------+----------------------------------------------------+
| ``LOADED``          | Instantiated and validated, registered in pluggy   |
+---------------------+----------------------------------------------------+
| ``ACTIVE``          | All contributions registered in the registry       |
+---------------------+----------------------------------------------------+
| ``FAILED``          | Error during load, validation, or registration     |
+---------------------+----------------------------------------------------+
| ``DISABLED``        | Explicitly deactivated by the user                 |
+---------------------+----------------------------------------------------+

Inspect plugin states::

    manager.get_plugin_state("topspin")      # PluginState.ACTIVE
    manager.get_active_plugins()             # ["topspin", ...]
    manager.get_failed_plugins()             # {"broken": "error msg"}
    manager.get_plugin_descriptor("topspin") # PluginDescriptor snapshot

Activation and deactivation::

    manager.deactivate_plugin("topspin")     # → marks DISABLED
    manager.activate_plugin("topspin")       # → marks ACTIVE

Deactivation is a lightweight state flag — no unloading or reimport
happens.  A disabled plugin is skipped if its entry point is
encountered again during discovery.


Lazy loading and optional dependencies
=======================================

Plugins should defer heavy imports to avoid slowing down SpectroChemPy
startup:

.. code-block:: python

    # ❌ Avoid: top-level import of a heavy library
    import numpy as np  # fine
    import torch        # ❌ heavy, loaded at startup


    # ✅ Prefer: deferred import inside operational methods
    class MyPlugin(SpectroChemPyPlugin):
        name = "mynet"
        version = "0.1.0"

        def register_readers(self) -> list[dict]:
            return [
                {
                    "name": "myformat",
                    "func": self._read_myformat,
                },
            ]

        def _read_myformat(self, path):
            from mynet import load_model  # deferred import
            ...

When a plugin fails to load (missing optional dependency,
``ImportError`` in constructor, etc.), the manager catches the
exception, marks the plugin as ``FAILED``, and continues.  Other
plugins and SpectroChemPy itself are unaffected::

    manager.get_failed_plugins()
    # → {"mynet": "No module named 'mynet'"}

This lets plugins declare optional dependencies freely without risking
startup crashes.


Test isolation
==============

Use fresh registry and manager instances per test to avoid state
leakage::

    from spectrochempy.plugins.registry import PluginRegistry
    from spectrochempy.plugins.manager import PluginManager


    def test_my_plugin():
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        # ... register plugin and assert on registry ...
