.. _install_adds:

Plugins and Example Data
========================

This page is for users who already have a working SpectroChemPy
installation (see :doc:`index`). Plugins add features for specific
techniques or file formats; install only those you need.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _install_adds_plugins:

Plugins
-------

.. list-table::
   :header-rows: 1

   * - Plugin
     - What it adds
   * - ``spectrochempy-nmr``
     - Bruker TopSpin reader and NMR processing
   * - ``spectrochempy-iris``
     - 2D-IRIS analysis
   * - ``spectrochempy-tensor``
     - Tensor decompositions (based on TensorLy)
   * - ``spectrochempy-hypercomplex``
     - Hypercomplex (quaternion) data
   * - ``spectrochempy-carroucell``
     - Reader for Carroucell experiments
   * - ``spectrochempy-perkinelmer``
     - Reader for PerkinElmer ``.sp`` files

Install a plugin in the same environment as SpectroChemPy, with the method
you used to install SpectroChemPy. Activate the environment first (for
instance with the shortcut created during installation).

.. tabs::

   .. tab:: Using uv (Recommended)

      .. code-block:: bash

         uv pip install spectrochempy-nmr           # one plugin
         uv pip install "spectrochempy[plugins]"    # all plugins

   .. tab:: Using conda / mamba

      .. code-block:: bash

         mamba install -c spectrocat spectrochempy-nmr    # one plugin

      Replace ``spectrochempy-nmr`` by the name of the plugin you need; to
      install several plugins, list their names in the same command.

   .. tab:: Using pip

      .. code-block:: bash

         python -m pip install spectrochempy-nmr           # one plugin
         python -m pip install "spectrochempy[plugins]"    # all plugins

   .. tab:: From sources

      From the plugin repository folder, with the environment active:

      .. code-block:: bash

         python -m pip install -e /path/to/plugin

      The NMR and hypercomplex plugins are maintained in standalone
      repositories; the other source plugins remain under ``plugins/``.

Each plugin is versioned independently. If you use a feature whose plugin is
missing, SpectroChemPy raises a ``MissingPluginError`` that tells you which
plugin to install. See :doc:`/userguide/plugins/index` for details on each
plugin.

Example Data
------------

The examples and tutorials of this documentation use example data files,
which are not included in the package to keep it small. **There is nothing to
install:** SpectroChemPy downloads them automatically the first time they are
needed, into the ``testdata`` folder of your SpectroChemPy settings folder
(``~/.spectrochempy``; on Windows, ``%USERPROFILE%\.spectrochempy``).

With conda / mamba, you can instead install them as a package, for instance
to work without an internet connection:

.. code-block:: bash

   mamba install -c spectrocat spectrochempy_data

Other Optional Packages
-----------------------

* **Jupyter**: installed with the ``[interactive]`` extra (pip, uv) or with
  ``mamba install jupyterlab`` (conda / mamba), as described in the
  installation guides.
* **xarray**: needed only to convert datasets to and from
  `xarray <https://xarray.dev>`_ objects
  (``uv pip install xarray``, ``python -m pip install xarray`` or
  ``mamba install xarray``).

If something goes wrong, see :doc:`../getting_help`.
