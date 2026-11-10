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

The standalone `NMR API reference
<https://www.spectrochempy.fr/projects/nmr/latest/reference/nmr_api.html>`_
provides the following entry points.

.. list-table::
   :header-rows: 1

   * - Public entry point
     - Purpose
   * - `scp.nmr.read() <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read.html>`_
     - Select a reader from the input format.
   * - `scp.nmr.read_topspin() <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_topspin.html>`_
     - Read Bruker TopSpin data.
   * - `scp.nmr.read_agilent() <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_agilent.html>`_
     - Read Agilent and Varian data.
   * - `scp.nmr.read_jeol() <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_jeol.html>`_
     - Read JEOL JDF data.
   * - `scp.nmr.read_simpson() <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.read_simpson.html>`_
     - Read SIMPSON simulations.
   * - `scp.nmr.Experiment <https://www.spectrochempy.fr/projects/nmr/latest/reference/generated/spectrochempy_nmr.Experiment.html>`_
     - Inspect and process NMR acquisitions.

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
