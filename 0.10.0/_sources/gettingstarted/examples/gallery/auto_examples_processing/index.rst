:orphan:

.. _examples-processing-index:

####################
Processing NDDataset
####################



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    </div>


Apodization
-----------

This section contains examples of how to apply apodization functions to NDDatasets.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we perform exponential window multiplication to apodize a NMR signal in the ti...">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/apodization/images/thumb/sphx_glr_plot_proc_em_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_apodization_plot_proc_em.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Exponential window multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we use sine bell or squared sine bell window multiplication to apodize a NMR s...">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/apodization/images/thumb/sphx_glr_plot_proc_sp_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_apodization_plot_proc_sp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sine bell and squared Sine bell window multiplication</div>
    </div>


.. raw:: html

    </div>


Baseline
--------

This section contains examples of how to correct the baseline of NDDatasets.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we perform a baseline correction of a 2D NDDataset interactively, using the mu...">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/baseline/images/thumb/sphx_glr_plot_baseline_correction_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_baseline_plot_baseline_correction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NDDataset baseline correction</div>
    </div>


.. raw:: html

    </div>


Denoising
-----------

This section contains examples of how to denoise NDDatasets.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we use the denoise method to remove the noise from a 2D Raman spectrum.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/denoising/images/thumb/sphx_glr_plot_denoising_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_denoising_plot_denoising.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Denoising a 2D Raman spectrum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we use the despike method to remove the noise from a Raman spectrum.">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/denoising/images/thumb/sphx_glr_plot_despike_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_denoising_plot_despike.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Removing cosmic ray spikes from a Raman spectrum</div>
    </div>


.. raw:: html

    </div>


filtering
---------

This section contains examples of how to filter NDDatasets.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Savitky-Golay and Whittaker-Eilers smoothing of a Raman spectrum">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/filtering/images/thumb/sphx_glr_plot_filter_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_filtering_plot_filter.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Savitky-Golay and Whittaker-Eilers smoothing of a Raman spectrum</div>
    </div>


.. raw:: html

    </div>


NMR processing (plugin-based)
-----------------------------

.. note::

    These examples require the official ``spectrochempy-nmr`` plugin.

    Install it with:

    .. code-block:: bash

        pip install spectrochempy[nmr]

    See :ref:`plugins` for more information.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/nmr/images/thumb/sphx_glr_plot_processing_cp_nmr_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_cp_nmr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Analysis CP NMR spectra</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/nmr/images/thumb/sphx_glr_plot_processing_nmr_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_nmr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Processing NMR spectra (slicing, baseline correction, peak picking, peak fitting)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Requires the official spectrochempy-nmr plugin. Install with: pip install spectrochempy[nmr].">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/nmr/images/thumb/sphx_glr_plot_processing_nmr_relax_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_processing_nmr_relax.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Processing Relaxation measurement</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to read Bruker TopSpin NMR files using the optional spectrochempy-nmr pl...">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/nmr/images/thumb/sphx_glr_plot_read_nmr_topspin_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_nmr_plot_read_nmr_topspin.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NMR: reading TopSpin files (plugin)</div>
    </div>


.. raw:: html

    </div>


Processing Raman datasets
-------------------------



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Processing RAMAN spectra">

.. only:: html

  .. image:: /gettingstarted/examples/gallery/auto_examples_processing/raman/images/thumb/sphx_glr_plot_processing_raman_thumb.png
    :alt:

  :ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_raman_plot_processing_raman.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Processing RAMAN spectra</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /gettingstarted/examples/gallery/auto_examples_processing/apodization/index.rst
   /gettingstarted/examples/gallery/auto_examples_processing/baseline/index.rst
   /gettingstarted/examples/gallery/auto_examples_processing/denoising/index.rst
   /gettingstarted/examples/gallery/auto_examples_processing/filtering/index.rst
   /gettingstarted/examples/gallery/auto_examples_processing/nmr/index.rst
   /gettingstarted/examples/gallery/auto_examples_processing/raman/index.rst

