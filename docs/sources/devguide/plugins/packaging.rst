.. _plugin-packaging:

=========================
Packaging a Plugin
=========================

This guide explains how to package and distribute a SpectroChemPy plugin.

Project structure
=================

A minimal plugin project looks like this::

    spectrochempy-myplugin/
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE
    ├── src/
    │   └── myplugin/
    │       └── __init__.py      # Plugin class + reader/writer functions
    ├── tests/
    │   ├── __init__.py
    │   └── test_plugin.py
    └── .github/
        └── workflows/
            └── test.yml         # CI configuration (optional)

``pyproject.toml``
==================

The entry point group ``spectrochempy.plugins`` tells SpectroChemPy how
to discover your plugin automatically:

.. code-block:: toml

    [build-system]
    requires = ["setuptools>=64"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "spectrochempy-myplugin"
    version = "0.1.0"
    description = "My SpectroChemPy plugin"
    requires-python = ">=3.11"
    dependencies = [
        "spectrochempy",
    ]

    [project.entry-points."spectrochempy.plugins"]
    myplugin = "myplugin:MyPlugin"

    [tool.setuptools.packages.find]
    where = ["src"]

    [tool.pytest.ini_options]
    testpaths = ["tests"]

Key points:

* The **entry point name** (``myplugin``) must match your plugin's
  ``name`` attribute.
* ``spectrochempy`` must be listed as a dependency.
* Use ``requires-python = ">=3.11"`` to match SpectroChemPy's minimum.

``__init__.py`` — the plugin class
==================================

Your plugin class must implement the
:class:`~spectrochempy.api.plugins.SpectroChemPyPlugin` protocol:

.. code-block:: python

    from spectrochempy.api.plugins import (
        CORE_PLUGIN_API_VERSION,
        SpectroChemPyPlugin,
    )


    class MyPlugin(SpectroChemPyPlugin):
        name = "myplugin"
        version = "0.1.0"
        description = "My SpectroChemPy plugin"
        spectrochempy_min_version = "0.8.0"
        PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION

        def register_readers(self) -> list[dict]:
            return [...]

See :ref:`plugin-architecture` for the full API reference.

Entry point discovery flow
==========================

1. User runs ``import spectrochempy`` (or calls ``scp.read_xxx``).
2. :class:`~spectrochempy.plugins.manager.PluginManager` scans
   ``importlib.metadata.entry_points(group="spectrochempy.plugins")``.
3. Each entry point is loaded, instantiated, and validated.
4. Declarative hooks (``register_readers()``, …) are collected.
5. Contributions are registered in the plugin registry.
6. The plugin is marked ``ACTIVE`` (or ``FAILED`` if an error occurs).

Distribution (PyPI)
===================

To publish on PyPI:

.. code-block:: bash

    pip install build twine
    python -m build
    twine upload dist/*

Name your package ``spectrochempy-<name>`` to make it discoverable.

CI / Testing
============

A recommended CI workflow tests against multiple Python versions::

    # .github/workflows/test.yml

    jobs:
      test:
        strategy:
          matrix:
            python-version: ["3.11", "3.12", "3.13"]
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: ["3.11", "3.12", "3.13"]  # set via CI matrix
          - name: Install SpectroChemPy
            run: |
              git clone https://github.com/spectrochempy/spectrochempy.git
              pip install -e ./spectrochempy
          - name: Install plugin
            run: pip install -e ".[dev]"
          - name: Run tests
            run: python -m pytest tests/ -v

Use the ``PluginTestHarness`` (see :ref:`plugin-testing`) for isolated
testing without touching the global registry.

See also
========

* :ref:`plugin-architecture` — Writing a plugin
* :ref:`plugin-testing` — Testing a plugin
