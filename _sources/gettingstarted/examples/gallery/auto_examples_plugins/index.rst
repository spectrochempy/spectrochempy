:orphan:

.. _examples-plugins-index:

###############################
Plugin-dependent functionality
###############################

This section contains examples that require optional SpectroChemPy plugins.
Official plugin examples also appear in their respective scientific sections,
so you can browse either by topic or by plugin.

.. _plugin-examples-list:

Examples requiring plugins
==========================

The examples listed below require one or more optional SpectroChemPy plugins.
They appear in the :ref:`main gallery <examples-index>` alongside core-only
examples, organised by scientific topic.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Example
     - Required plugin
     - Gallery section
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_analysis_a_decomposition_plot_iris_intro.py`
     - ``spectrochempy-iris``
     - :ref:`Decomposition <examples-analysis-decomposition-index>`
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_core_c_importer_plot_read_nmr_from_bruker.py`
     - ``spectrochempy-nmr``
     - Import/export
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_apodization_plot_proc_em.py`
     - ``spectrochempy-nmr``
     - Processing / apodization
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_apodization_plot_proc_sp.py`
     - ``spectrochempy-nmr``
     - Processing / apodization
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_read_nmr_topspin.py`
     - ``spectrochempy-nmr``
     - :ref:`NMR processing <examples-processing-nmr-index>`
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_nmr.py`
     - ``spectrochempy-nmr``
     - :ref:`NMR processing <examples-processing-nmr-index>`
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_cp_nmr.py`
     - ``spectrochempy-nmr``
     - :ref:`NMR processing <examples-processing-nmr-index>`
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_nmr_relax.py`
     - ``spectrochempy-nmr``
     - :ref:`NMR processing <examples-processing-nmr-index>`
   * - :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plot_read_perkinelmer.py`
     - ``spectrochempy-perkinelmer``
     - Import/export

See :ref:`plugins` for general plugin installation instructions.




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

IRIS plugin
-----------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example introduces the 2D-IRIS analysis provided by the optional spectrochempy-iris plugin.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-iris/images/thumb/sphx_glr_plot_spectrochempy_iris__analysis__a_decomposition__plot_iris_intro_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-iris_plot_spectrochempy_iris__analysis__a_decomposition__plot_iris_intro.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IRIS: 2D-IRIS analysis (plugin)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

NMR plugin
----------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we load a NMR dataset (in the Bruker format) and plot it.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__core__c_importer__plot_read_nmr_from_bruker_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__core__c_importer__plot_read_nmr_from_bruker.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading of experimental NMR data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we perform exponential window multiplication to apodize a NMR signal in the time domain.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__apodization__plot_proc_em_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__apodization__plot_proc_em.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Exponential window multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we use sine bell or squared sine bell window multiplication to apodize a NMR signal in the time domain.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__apodization__plot_proc_sp_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__apodization__plot_proc_sp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sine bell and squared Sine bell window multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__nmr__plot_processing_cp_nmr_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__nmr__plot_processing_cp_nmr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Analysis CP NMR spectra</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__nmr__plot_processing_nmr_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__nmr__plot_processing_nmr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Processing NMR spectra (slicing, baseline correction, peak picking, peak fitting)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__nmr__plot_processing_nmr_relax_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__nmr__plot_processing_nmr_relax.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Processing Relaxation measurement</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to read Bruker TopSpin NMR files using the optional spectrochempy-nmr plugin.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/images/thumb/sphx_glr_plot_spectrochempy_nmr__processing__nmr__plot_read_nmr_topspin_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-nmr_plot_spectrochempy_nmr__processing__nmr__plot_read_nmr_topspin.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NMR: reading TopSpin files (plugin)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

PerkinElmer plugin
------------------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to read a PerkinElmer .sp binary IR file using the optional spectrochempy-perkinelmer plugin.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-perkinelmer/images/thumb/sphx_glr_plot_spectrochempy_perkinelmer__plot_read_perkinelmer_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_plugins_spectrochempy-perkinelmer_plot_spectrochempy_perkinelmer__plot_read_perkinelmer.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reading a PerkinElmer SP file (plugin)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-iris/index.rst
   /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-nmr/index.rst
   /gettingstarted/examples/gallery/auto_examples_plugins/spectrochempy-perkinelmer/index.rst

