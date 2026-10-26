.. _plugin_public_api_reference:

###########################
Plugin public API reference
###########################

This page collects the generated public API pages owned by official
SpectroChemPy plugins.

Workflow-oriented plugin guides remain in :doc:`/userguide/plugins/index`.

The generated entries below target the public objects exported by each
standalone plugin package. The ``scp.nmr.*`` and equivalent namespace routes
remain the recommended user-facing aliases, but they are lazy proxies and are
not suitable as autodoc targets.

NMR plugin
==========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy_nmr.read
    spectrochempy_nmr.read_topspin
    spectrochempy_nmr.read_agilent
    spectrochempy_nmr.Experiment

PerkinElmer plugin
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.perkinelmer.read
    spectrochempy.perkinelmer.read_perkinelmer

Carroucell plugin
=================

Recommended public entry point:
:func:`spectrochempy.carroucell.read`.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.carroucell.read_carroucell

IRIS plugin
===========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.iris.IRIS

Tensor plugin
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.tensor.CP

Hypercomplex plugin
===================

The recommended public API is the ``dataset.hyper`` accessor.

All hypercomplex operations are accessed through ``dataset.hyper``:

.. list-table::
   :header-rows: 1

   * - Operation
     - Example
   * - Convert to quaternion
     - ``dataset.hyper.set_quaternion(inplace=True)``
   * - Check type
     - ``dataset.hyper.is_quaternion``
   * - Extract RR component
     - ``dataset.hyper.RR``
   * - Extract RI component
     - ``dataset.hyper.component("RI")``
   * - Extract IR component
     - ``dataset.hyper.IR``
   * - Extract II component
     - ``dataset.hyper.II``

For workflow-oriented explanations and examples, see
:doc:`/userguide/plugins/hypercomplex`.
