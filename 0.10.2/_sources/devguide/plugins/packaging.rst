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
        "spectrochempy>=0.9,<0.10",
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
   range, e.g. ``spectrochempy>=0.10,<0.11``.
* Use ``requires-python = ">=3.11"`` to match SpectroChemPy's minimum.
* Official plugins in the monorepo use a static ``version`` field.
  ``setuptools_scm`` is not used because plugin and core tags share the
  same Git repository, which would cause version collisions.
* Official plugins must add the Trove classifier
  ``Framework :: SpectroChemPy :: Official Plugin`` in their
  ``pyproject.toml`` classifiers list.  This classifier is the single
  source of truth for CI workflows (publishing, testing, documentation,
  release validation) — any plugin with this classifier is automatically
  picked up by all CI pipelines.  Adding a new official plugin requires
  **no edits** to any workflow file.

Local editable development
==========================

During local development, install the core package and each plugin in
editable mode.  This exercises the same entry-point discovery mechanism
used by PyPI wheels:

.. code-block:: bash

    pip install -e .
    python -m spectrochempy.ci.install_plugins --editable all
    pip install -e plugins/spectrochempy-cantera  # experimental, not auto-discovered

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
    pip install "spectrochempy[tensor]"
    pip install "spectrochempy[plugins]"

The ``nmr`` extra installs the NMR plugin, currently including the TopSpin
reader and its NMR-specific dependencies.  The ``plugins`` extra installs the
current official plugin set.

Experimental plugins such as ``spectrochempy-cantera`` are **not** included
in aggregate extras and must be installed directly::

    pip install spectrochempy-cantera

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
        spectrochempy_min_version = "0.9.0"
        PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION

        def register_readers(self) -> list[dict]:
            return [...]

See :ref:`plugin-architecture` for the runtime architecture overview and
:ref:`plugin-author-guide` for the author-facing quick-start guide.

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

* A core release tag such as ``spectrochempy-v0.9.0`` publishes the core wheel only;
  it does **not** trigger plugin uploads.
* A plugin release tag such as ``spectrochempy-nmr-v0.1.1`` triggers the
  ``publish_plugins.yml`` and ``build_package.yml`` workflows for that
  plugin only.
* The workflow uses ``skip-existing: true`` on PyPI so an already-published
  version never causes a hard failure.

Bumping a plugin version
------------------------

Plugin versions are declared **statically** in the monorepo (``setuptools-scm``
is not used for plugins because plugin and core tags share the same Git
repository, which leads to version collisions).

Stable plugin tags are named ``spectrochempy-<plugin>-vX.Y.Z``.  Development
builds derive their temporary version from the latest plugin tag and the number
of commits that touched release-relevant files still differing from that tag.
Those files are the plugin ``src/`` tree and published metadata such as
``pyproject.toml``, ``recipe.yaml``, ``README.md``, ``LICENSE``, and
``MANIFEST.in``.  Tests, documentation-only changes, CI changes, and CRLF-only
differences do not mark a plugin as modified.  For example, if the latest
``spectrochempy-nmr`` tag is ``spectrochempy-nmr-v0.1.3`` and 12 relevant
commits remain, CI builds ``spectrochempy-nmr`` as ``0.1.4.dev12``.

The dev version uses the next patch version, not the current patch version:
``0.1.4.dev12`` is newer than ``0.1.3``, whereas ``0.1.3.dev12`` is older than
``0.1.3`` under PEP 440.  CI injects this version into ``pyproject.toml``,
``recipe.yaml``, and the plugin class only inside the workflow workspace; no
version bump is committed for development builds.

The recommended way to release a plugin is through the
``release_plugin.yml`` workflow:

1. Optionally run **Actions → Check plugin release status** first to inspect
   which official plugins changed since their last release tag.
2. Go to **Actions → Release an official plugin** in the GitHub UI.
3. Click **Run workflow** from the ``master`` branch.
4. Enter:
   - Plugin name: ``spectrochempy-nmr``
   - Version: ``0.1.1``
5. The workflow bumps ``pyproject.toml``, ``recipe.yaml``, and the plugin
   ``__init__.py`` version string, commits to ``master``, and creates
   the release tag ``spectrochempy-nmr-v0.1.1`` automatically.
6. The tag triggers CI which builds and publishes the wheel to PyPI and
   Anaconda.org.

Manual fallback::

    # 1. Bump version in plugins/<name>/pyproject.toml
    # 2. Bump version in plugins/<name>/recipe.yaml (conda)
    # 3. Commit and push:
    git add plugins/<name>/pyproject.toml plugins/<name>/recipe.yaml \
        plugins/<name>/src/spectrochempy_<name>/__init__.py
    git commit -m "Bump spectrochempy-<name> to 0.1.1"
    git push upstream master

    # 4. Create and push the release tag:
    git tag spectrochempy-<name>-v0.1.1
    git push upstream spectrochempy-<name>-v0.1.1

The tag is a pure CI trigger; the actual package version comes from
``pyproject.toml`` (pip) and ``recipe.yaml`` (conda).

Distribution (conda)
====================

Official plugins with a ``recipe.yaml`` in their root are built and
published to Anaconda.org automatically by ``build_package.yml``.

* **Stable builds** (releases with a ``spectrochempy-<plugin>-vX.Y.Z`` tag)
  are uploaded to the ``main`` label:

  .. code-block:: bash

      mamba install -c spectrocat -c conda-forge spectrochempy-nmr

* **Development builds** on ``master`` use generated versions such as
  ``0.1.4.dev12`` and upload to ``spectrocat/label/dev`` only when the
  computed plugin status reports release-relevant changes.  Pull requests build
  the packages for validation but do not publish them.

  Stable plugin release uploads remain on the ``main`` label.

Plugin recipes should declare a bounded dependency on the core package,
e.g. ``spectrochempy >=0.9,<0.10``.

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

* :ref:`plugin-architecture` — Plugin architecture
* :ref:`plugin-author-guide` — Plugin author guide
* :ref:`plugin-testing` — Testing a plugin
