.. _plugin_public_api_reference:

###########################
Plugin public API reference
###########################

This page collects the generated public API pages owned by official
SpectroChemPy plugins.

Workflow-oriented plugin guides remain in :doc:`/userguide/plugins/index`.

NMR plugin
==========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.nmr.read
    spectrochempy.nmr.Experiment

PerkinElmer plugin
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.perkinelmer.read

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
