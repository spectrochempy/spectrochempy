Fourier transformation
======================

For NMR data, follow the plugin's
`1D processing guide <https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_1d.html>`_
and `2D processing guide <https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_2d.html>`_.

The core provides :func:`~spectrochempy.fft` and :func:`~spectrochempy.ifft`
for datasets with calibrated coordinates. Since SpectroChemPy 1.2.1,
phase correction is a separate step. Use :func:`~spectrochempy.pk` on
a complex spectrum after the FFT when needed. For FTIR interferograms, see
:doc:`interferogram`.
