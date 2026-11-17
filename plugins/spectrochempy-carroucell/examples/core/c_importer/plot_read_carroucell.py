# ruff: noqa: D205, D400, D415
"""
Plot a sample series from a Carroucell experiment
=================================================

Use the optional ``spectrochempy-carroucell`` plugin to read a directory and
plot spectra from one sample. Install it with ``pip install spectrochempy-carroucell``.

See :ref:`import-carroucell-ir` for directory layout and temperature logs.
"""

import spectrochempy as scp

series = scp.carroucell.read("irdata/carroucell_samp", spectra=(1, 3))
sample = next(dataset for dataset in series if dataset.name == "3-SILICA")
_ = sample.plot_lines(title="Silica sample")
