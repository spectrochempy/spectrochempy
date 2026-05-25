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
        "spectrochempy>=0.8,<0.9",
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
* ``spectrochempy`` must be listed as a dependency with a compatibility
  range, e.g. ``spectrochempy>=0.8,<0.10``.
* Use ``requires-python = ">=3.11"`` to match SpectroChemPy's minimum.
* Official plugins in the monorepo use a static ``version`` field.
  When a plugin is moved to its own repository, ``setuptools_scm``
  is recommended so that tags such as ``spectrochempy-nmr-v0.1.1``
  drive the published version automatically.

Local editable development
==========================

During local development, install the core package and each plugin in
editable mode.  This exercises the same entry-point discovery mechanism
used by PyPI wheels:

.. code-block:: bash

    pip install -e .
    pip install -e plugins/spectrochempy-nmr
    pip install -e plugins/spectrochempy-iris
    pip install -e plugins/spectrochempy-cantera

The bundled plugins can also be installed with the helper:

.. code-block:: bash

    python -m spectrochempy.ci.install_plugins --editable all

After installation, SpectroChemPy discovers the plugins automatically via
the ``spectrochempy.plugins`` entry point group.

Installing from SpectroChemPy extras
====================================

Users can install official plugins through SpectroChemPy extras:

.. code-block:: bash

    pip install "spectrochempy[nmr]"
    pip install "spectrochempy[iris]"
    pip install "spectrochempy[cantera]"
    pip install "spectrochempy[plugins]"

The ``nmr`` extra installs the NMR plugin, currently including the TopSpin
reader and its NMR-specific dependencies.  The ``plugins`` extra installs the
current official plugin set.

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

Official plugins that still live in the SpectroChemPy monorepo are published
from the root workflow:

.. code-block:: text

    .github/workflows/publish_plugins.yml

GitHub Actions only executes workflows from the repository root, so workflows
stored inside ``plugins/<plugin>/.github/workflows/`` are templates only while
the plugin remains in the monorepo.  If a plugin is later moved to its own
repository, copy the template workflow to that repository's root
``.github/workflows/`` directory.

The monorepo workflow uses PyPI Trusted Publishing.  Each plugin distribution
must therefore have its own PyPI/TestPyPI trusted publisher configured with the
matching project name, for example ``spectrochempy-nmr``.

For a manual local upload:

.. code-block:: bash

    pip install build twine
    python -m build
    twine upload dist/*

Name your package ``spectrochempy-<name>`` to make it discoverable.

Release policy
==============

Plugins are released **independently** from the core package.

* A core release tag such as ``spectrochempy-v0.8.3`` publishes the core
  wheel only; it does **not** trigger plugin uploads.
* A plugin release tag such as ``spectrochempy-nmr-v0.1.1`` triggers the
  ``publish_plugins.yml`` workflow for that plugin only.
* The workflow uses ``skip-existing: true`` on PyPI so an already-published
  version never causes a hard failure.

Tag-based versioning
--------------------

Official plugins use `setuptools-scm <https://setuptools-scm.readthedocs.io/>`_
to derive their version from Git tags.  The tag convention is::

    spectrochempy-<plugin>-v<version>

Examples::

    spectrochempy-nmr-v0.1.1
    spectrochempy-iris-v0.2.0
    spectrorochempy-cantera-v0.1.0

When a matching tag is pushed, the CI workflow builds the wheel with that
version and publishes it.  If no tag exists, ``fallback_version`` (set in
``pyproject.toml``) is used for local or dev builds.

To bump a plugin version in the monorepo::

    # 1. Update the fallback_version in plugins/<name>/pyproject.toml
    # 2. Update the version in plugins/<name>/recipe.yaml (conda)
    # 3. Commit, tag, and push:
    git tag spectrochempy-nmr-v0.1.1
    git push upstream spectrochempy-nmr-v0.1.1

Distribution (conda)
====================

Official plugins with a ``recipe.yaml`` in their root are built and
published to Anaconda.org automatically by ``build_package.yml``.

* **Dev builds** (pushes, PRs) are uploaded to the ``dev`` label:

  .. code-block:: bash

      mamba install -c spectrocat/label/dev -c conda-forge spectrochempy-nmr

* **Stable builds** (releases) are uploaded to the ``main`` label:

  .. code-block:: bash

      mamba install -c spectrocat -c conda-forge spectrochempy-nmr

Plugin recipes should declare a bounded dependency on the core package,
e.g. ``spectrochempy >=0.8,<0.10``.

The ``plugin-template`` directory is excluded from discovery; it is a
developer scaffold and must never be published.

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
