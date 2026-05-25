.. _plugin-author-guide:

===================
Plugin author guide
===================

A SpectroChemPy plugin is a Python package that exposes an entry point in the
``spectrochempy.plugins`` group and returns declarative contributions.

Minimal structure
=================

::

    myplugin/
      pyproject.toml
      src/myplugin/__init__.py
      src/myplugin/readers.py

The entry point points to a plugin class:

.. code-block:: toml

    [project.entry-points."spectrochempy.plugins"]
    myplugin = "myplugin:MyPlugin"

The plugin declares contributions:

.. code-block:: python

    from spectrochempy.api.plugins import SpectroChemPyPlugin

    from .readers import read_myformat


    class MyPlugin(SpectroChemPyPlugin):
        name = "myplugin"
        version = "0.1.0"

        def register_readers(self) -> list[dict]:
            return [
                {
                    "name": "myformat",
                    "func": read_myformat,
                    "description": "Read MyFormat files",
                    "extensions": [".myf"],
                },
            ]

Readers and writers
===================

Readers create datasets and belong under ``scp.<plugin>`` or compatibility
``scp.read_<format>`` aliases. They should not be dataset accessors.

Use plugin handlers for format-specific filename inference:

.. code-block:: python

    def register_handlers(self) -> dict:
        return {
            "importer.resolve_directory_target": resolve_directory_target,
            "importer.infer_filetype_key": infer_filetype_key,
        }

Accessors
=========

Dataset accessors are for operations that act on an existing
``NDDataset``:

.. code-block:: python

    def register_accessors(self) -> list[dict]:
        return [
            {
                "namespace": "myplugin",
                "name": "normalize",
                "func": normalize_dataset,
                "description": "Normalize a dataset using MyPlugin rules",
            },
        ]

Handlers
========

Handlers are named extension points used when the core must delegate behavior
without importing plugin-specific types or conventions:

.. code-block:: python

    def register_handlers(self) -> dict:
        return {
            "ndmath.execution_branch": execution_branch,
            "ndmath.execute": execute_numeric_branch,
            "importer.remote_download_target": remote_download_target,
        }

Numeric backends and unit contexts
==================================

Use numeric backend handlers when plugin data needs a non-standard execution
path, such as quaternion arrays. Use unit contexts when unit conversion needs
domain metadata, such as an acquisition frequency.

Official and third-party plugins
================================

Official plugins are maintained with SpectroChemPy and may receive coordinated
docs, examples, tests, and temporary compatibility aliases. Third-party plugins
use the same public API but should document their own namespaces and avoid
depending on private ``spectrochempy.plugins`` internals.

See also :ref:`plugin-architecture`, :ref:`plugin-accessors`,
:ref:`plugin-numeric-backends`, and :ref:`plugin-unit-contexts`.
