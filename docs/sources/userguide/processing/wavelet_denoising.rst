Wavelet denoising
=================

Install the optional PyWavelets dependency with
``pip install "spectrochempy[wavelets]"``, then apply
:class:`~spectrochempy.Filter` along a spectral dimension.

.. code-block:: python

   import spectrochempy as scp

   filtered = scp.Filter(method="wavelet").transform(spectrum, dim="x")

By default, the filter uses Symlet-4, symmetric extension, and up to three
decomposition levels. It estimates one soft threshold per spectrum from
the median absolute deviation of the finest detail coefficients. Use
``wavelet``, ``wavelet_level``, or ``wavelet_threshold`` to change these settings.
An explicit threshold uses the signal units.

The transform operates in sample space and requires real, finite, unmasked
data. The result retains the input shape, coordinates, and units. The
processing history records the effective settings.

The :doc:`smoothing comparison </gettingstarted/examples/gallery/auto_examples_processing/filtering/plot_compare_spectral_smoothers>`
shows the effects on noise, peak position, height, and width.
