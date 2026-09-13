.. _experimental-plugins:

====================
Experimental plugins
====================

Some plugins in the SpectroChemPy repository are considered **experimental**.
They are **not** officially supported, **not** included in aggregate extras, and
**not** auto-published by the official release workflows.

What "experimental" means
=========================

* APIs may change without deprecation warnings or migration guides.
* Examples may be incomplete or missing.
* Installation is manual.
* Support is limited; they are provided mainly for developers and early testers.
* Some experimental plugins may eventually become official, but there is no
  defined timeline.


Installing experimental plugins
===============================

Experimental plugins are installed directly from PyPI or from source:

.. code-block:: bash

    pip install spectrochempy-cantera

or from the repository:

.. code-block:: bash

    pip install -e plugins/spectrochempy-cantera

Because they are not part of ``spectrochempy[plugins]``, installing the core
package extras will **not** pull them in automatically.

Third-party plugins
===================

External developers can publish third-party plugins that use the same
``spectrochempy.plugins`` entry-point mechanism. Third-party plugins are
discovered automatically once installed, just like official and experimental
plugins. They are not listed in the official or experimental registries.

Cantera (experimental)
======================

.. warning::
    **EXPERIMENTAL** — The ``spectrochempy-cantera`` plugin is not officially
    supported. Its API is subject to change without notice.

The ``spectrochempy-cantera`` plugin provides a plug-flow reactor (PFR) model
based on Cantera. It is kept as an experimental plugin because the simulation
APIs are still evolving and have not yet reached the stability required for
official support.

Installation
------------

.. code-block:: bash

    pip install spectrochempy-cantera

Usage
-----

Once installed, the plugin is discovered automatically:

.. code-block:: python

    import spectrochempy as scp

    PFR = scp.cantera.PFR

Or import directly from the plugin package:

.. code-block:: python

    from spectrochempy_cantera import PFR

Scope
-----

The current public scope is limited to the ``PFR`` class. Earlier experimental
APIs such as ``scp.cantera.equilibrium`` and ``scp.cantera.reactor_profile``
have been removed.

For questions or contributions, see the plugin source at
``plugins/spectrochempy-cantera/``.
