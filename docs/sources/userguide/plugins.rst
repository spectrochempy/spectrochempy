.. _plugins:

========
Plugins
========

SpectroChemPy uses plugins to keep the project modular while preserving a
simple scientific workflow. Official plugins remain part of the SpectroChemPy
ecosystem: they provide domain-specific readers, analyses, and simulations, but
they are discovered automatically once installed.

For most users, this means:

* standard infrared workflows remain natural and easy to use;
* no explicit plugin-loading command is needed in normal code;
* optional or heavier dependencies are imported only when the corresponding
  feature is used;
* specialized domains can evolve without making the technical core heavier.

Pluginization is therefore mostly an internal modularization strategy. It is
not intended to make established SpectroChemPy workflows feel removed or harder
to access.

.. _plugins-install:

Installing a Plugin
===================

Plugins are distributed as separate packages on PyPI. Install them with
``pip``:

.. code-block:: bash

    pip install spectrochempy[nmr]      # NMR / TopSpin reader
    pip install spectrochempy[iris]     # 2D-IRIS analysis
    pip install spectrochempy[cantera]  # PFR reactor simulation

Or install a specific plugin directly:

.. code-block:: bash

    pip install spectrochempy-nmr
    pip install spectrochempy-iris
    pip install spectrochempy-cantera

.. _plugins-usage:

Using a Plugin
==============

Once installed, the plugin registers itself automatically when you import
SpectroChemPy. There is no ``load_plugin(...)`` step for normal use.

.. _plugins-lazy:

.. note::

    Plugin optional dependencies (such as ``osqp``, ``cantera``, ``numpy-quaternion``,
    ``nmrglue``) are **not imported** when you run ``import spectrochempy``.
    They are loaded lazily, only when you actually call a plugin function.
    This keeps startup fast and avoids unnecessary imports.

Plugin functions and classes are available from package-level namespaces,
and dataset-bound operations are available from dataset accessors:

.. code-block:: python

    import spectrochempy as scp

    # NMR plugin: read TopSpin files
    ds = scp.nmr.read_topspin("path/to/fid")

    # Cantera plugin: access the PFR simulation callable
    PFR = scp.cantera.PFR

    # IRIS plugin: use the modern namespaced analysis class
    iris_analysis = scp.iris.IRIS()

    # IRIS plugin: build an IRIS kernel from an existing dataset
    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

The distinction is:

* use ``scp.<plugin>.*`` for readers, object creation, simulations, and other
  package-level workflows;
* use ``dataset.<plugin>.*`` only for operations that act on an existing
  :class:`~spectrochempy.NDDataset`.

You can also import directly from a plugin namespace:

.. code-block:: python

    from spectrochempy.iris import IRIS, IrisKernel
    from spectrochempy.nmr import read_topspin
    from spectrochempy.cantera import PFR

    import spectrochempy.iris as iris
    iris.IRIS()

All three forms - ``scp.iris.IRIS``, ``from spectrochempy import iris; iris.IRIS``,
and ``from spectrochempy.iris import IRIS`` - are supported and preserve the
:ref:`lazy loading <plugins-lazy>` behaviour described below.

For backward compatibility, some former top-level APIs remain available as thin
aliases. For example, ``scp.read_topspin(...)`` delegates to
``scp.nmr.read_topspin(...)`` when the NMR plugin is installed.

Some official plugins may also provide explicit root-level compatibility aliases.
For example, ``scp.IRIS`` delegates to ``scp.iris.IRIS`` and emits a
``FutureWarning`` when accessed. New code and examples should prefer the
namespaced API, such as ``scp.iris.IRIS``. Compatibility aliases are intentionally
limited and may disappear in a future release.

If a plugin is not installed, the corresponding official optional feature gives
a clear installation hint. For example:

.. code-block:: pycon

    >>> import spectrochempy as scp
    >>> scp.read_topspin("path/to/fid")
    MissingPluginError: The 'read_topspin' feature requires the optional plugin
    'spectrochempy-nmr'. Install it with: pip install spectrochempy[nmr]

.. _plugins-list:

Official and Third-Party Plugins
================================

Official plugins are maintained with the SpectroChemPy project and documented
alongside the main package. Third-party plugins can use the same discovery
mechanism, but they declare and document their own contributions.

Installed plugins are discovered through Python entry points. Missing official
plugins are known only well enough to produce clear installation hints when a
feature is accessed.

Available Official Plugins
==========================

.. list-table::
   :header-rows: 1

   * - Plugin
     - Package
     - Provides
   * - NMR
     - ``spectrochempy-nmr``
      - ``scp.nmr.read_topspin`` (Bruker TopSpin reader),
        NMR processing utilities
   * - IRIS
     - ``spectrochempy-iris``
     - 2D-IRIS analysis
       (:class:`~spectrochempy_iris.IRIS`,
       :class:`~spectrochempy_iris.IrisKernel`)
   * - Cantera
     - ``spectrochempy-cantera``
     - ``scp.cantera.PFR`` plug flow reactor simulation callable

.. _plugins-dev-install:

Installing plugins in development mode
=======================================

If you are working on the SpectroChemPy source (for example from the monorepo
with the ``plugins`` branch), you can install the plugins in **editable mode**
with ``pip``:

.. code-block:: bash

    pip install -e plugins/spectrochempy-nmr
    pip install -e plugins/spectrochempy-iris
    pip install -e plugins/spectrochempy-cantera

Editable mode (``-e``) lets you modify the plugin source and test changes
without reinstalling.

.. _plugins-developer:

Developing a Plugin
===================

See the :ref:`plugin-architecture` section for the plugin API documentation,
packaging guide, and a full reference example (``spectrochempy-nmr``).

Related pages
=============

.. toctree::
   :maxdepth: 1

   plugins_official
   plugins_hypercomplex
   plugins_examples
   plugins_roadmap
