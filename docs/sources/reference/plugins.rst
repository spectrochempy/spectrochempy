.. _plugin_public_api_reference:

###########################
Plugin public API reference
###########################

This page provides links and reference pages for the public APIs of
official SpectroChemPy plugins.

Workflow-oriented plugin guides remain in :doc:`/userguide/plugins/index`.
For file formats and their public reader entry points, see the
:doc:`reader catalog </userguide/importexport/readers>`.

NMR plugin
==========

Use ``scp.nmr.read()`` to read NMR data. The standalone
`NMR API reference
<https://spectrochempy-nmr.readthedocs.io/latest/reference/nmr_api.html>`_
documents the readers, ``Experiment``, and interoperability functions.

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
