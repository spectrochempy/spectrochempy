# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Fit two overlapping Gaussian bands
==================================

Fit two overlapping bands in a synthetic spectrum and inspect the
fitted components and residuals.
See the :doc:`fitting tutorial </userguide/analysis/fitting>` for model
scripts, parameter bounds, and fitting overlapping experimental bands.
"""

from contextlib import redirect_stdout
from io import StringIO

import spectrochempy as scp

x = scp.Coord.linspace(1500.0, 1700.0, 201, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(x.data, ampl=20.0, pos=1588.0, width=30.0)
band_b = scp.gaussian(x.data, ampl=16.0, pos=1610.0, width=34.0)
signal = band_a + band_b
noise = scp.normal(scale=0.01, size=signal.shape, seed=42)
spectrum = scp.NDDataset(signal + noise.data, coordset=[x], title="absorbance")

script = """
MODEL: band_a
shape: gaussianmodel
$ ampl: 18, 0, none
$ pos: 1585, 1550, 1599
$ width: 35, 1, 80

MODEL: band_b
shape: gaussianmodel
$ ampl: 14, 0, none
$ pos: 1615, 1599, 1650
$ width: 30, 1, 80
"""
model = scp.Optimize(script=script, method="least_squares")
with redirect_stdout(StringIO()):
    _ = model.fit(spectrum)
ax = model.plot_merit(
    exp_label="Data", calc_label="Sum of bands", resid_label="Residual"
)
legend = ax.get_legend()
handles = list(legend.legend_handles)
labels = [entry.get_text() for entry in legend.get_texts()]
for component, label in zip(model.components.data, ("Band A", "Band B"), strict=True):
    (line,) = ax.plot(x.data, component, linestyle=":")
    handles.append(line)
    labels.append(label)
_ = ax.legend(handles, labels)
