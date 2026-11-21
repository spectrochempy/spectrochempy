"""
Experimental Wang-style distribution-classified baseline, NumPy/SciPy only.

Based on Wang et al., Anal. Chem. 2013, 85, 1231-1239,
DOI 10.1021/ac303233c, Supporting Information S5-S7. Uses averaged segment
endpoints to resolve the Y/R ambiguity in S7, population standard deviations
and the positive-estimate noise filter qualified in nmrglue-ng PR 284.
Diffusion-edited and resampling extensions are not implemented.

Adapted from nmrglue.process.proc_bl (nmrglue-ng PR 284), with amplitude
normalization and dataset integration added here. The original BSD notice
for the adapted portions follows.

Copyright Notice and Statement for the nmrglue Project


Copyright (c) 2010-2015 Jonathan J. Helmus
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:


a. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.


b. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.


c. Neither the name of the author nor the names of contributors may
   be used to endorse or promote products derived from this software
   without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.signal import convolve


def _clipped_mean(values, radius):
    radius = min(radius, values.size - 1)
    weights = np.full(2 * radius + 1, 1.0 / (2 * radius + 1))
    return convolve(values, weights, mode="same") / convolve(
        np.ones(values.size), weights, mode="same"
    )


def _distribution_baseline(values, width):
    """Return the baseline of one finite real trace, without changing it."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values))
    if scale == 0:
        return np.zeros_like(values)
    normalized = values / scale
    origin = normalized[0]
    centered = normalized - origin
    size = values.size
    radius = min(width, size - 1)
    deviations = np.empty(size)
    span = 2 * radius + 1
    if span <= size:
        windows = np.lib.stride_tricks.sliding_window_view(centered, span)
        deviations[radius : size - radius] = np.std(windows, axis=-1)
        edges = list(range(radius)) + list(range(size - radius, size))
    else:
        edges = range(size)
    for i in edges:
        deviations[i] = np.std(centered[max(0, i - radius) : min(size, i + radius + 1)])
    estimates = deviations.copy()
    sigma = np.median(estimates)
    while sigma > 0:
        estimates = estimates[(estimates > 0) & (estimates <= 2 * sigma)]
        updated = np.median(estimates)
        if updated / sigma >= 0.999:
            sigma = updated
            break
        sigma = updated
    signal = np.zeros(size, dtype=bool)
    for index in np.flatnonzero(deviations > 1.1 * sigma):
        signal[max(0, index - 3) : min(size, index + 4)] = True
    if signal.all():
        raise ValueError("distribution baseline has no noise-only support")
    boundaries = np.flatnonzero(np.diff(np.r_[False, signal, False]))
    temporary = centered.copy()
    for start, stop in boundaries.reshape(-1, 2):
        end = stop - 1
        left = np.mean(centered[max(0, start - 7) : min(size, start + 8)])
        right = np.mean(centered[max(0, end - 7) : min(size, end + 8)])
        temporary[start:stop] = np.linspace(left, right, stop - start)
    baseline = _clipped_mean(temporary, 60) + origin
    return np.clip(baseline, normalized.min(), normalized.max()) * scale
