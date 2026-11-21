"""
Second-difference arPLS and asPLS, with banded NumPy/SciPy solves.

Weighting and iteration adapted from pybaselines 1.2.1 (Donald Erb).
arPLS: Baek et al., Analyst 2015, 140, 250–257, DOI 10.1039/C4AN01061B.
asPLS: Zhang et al., Spectroscopy Letters 2020, 53, 222–233,
DOI 10.1080/00387010.2020.1730908. The default asymmetric coefficient is
0.5, following the pybaselines variant, rather than the published 2.

BSD 3-Clause License

Copyright (c) 2021 Donald Erb
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import expit


def _whittaker_baseline(values, model, lamb, tol, max_iter, asymmetric_coef=0.5):
    """Return a baseline and whether the relative weight tolerance was reached."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values))
    if scale == 0:
        return values.copy(), True
    if 1e-100 <= scale <= 1e100:
        scale = 1.0
    y = values / scale
    precision = 64 * np.finfo(float).eps * np.max(np.abs(y))
    if np.max(np.abs(np.diff(y, n=2))) <= precision:
        return values.copy(), True

    n = y.size
    penalty = np.zeros((5, n))
    for row, left in enumerate((1.0, -2.0, 1.0)):
        for col, right in enumerate((1.0, -2.0, 1.0)):
            penalty[2 + row - col, col : col + n - 2] += lamb * left * right
    weights = np.ones(n)
    alpha = np.ones(n)
    converged = False
    for _ in range(max_iter):
        system = penalty.copy()
        if model == "aspls":
            for band, offset in enumerate(range(-2, 3)):
                first, last = max(0, -offset), min(n, n - offset)
                system[band, first:last] *= alpha[first + offset : last + offset]
        system[2] += weights
        baseline = solve_banded((2, 2), system, weights * y)
        residual = y - baseline
        negative = residual[residual < 0]
        if np.max(np.abs(residual)) <= precision:
            converged = True
            break
        if negative.size < 2:
            break
        deviation = negative.std(ddof=1)
        if deviation <= np.finfo(float).eps * np.max(np.abs(y)):
            break
        if model == "arpls":
            updated = expit(
                -(2 / deviation) * (residual - (2 * deviation - negative.mean()))
            )
        else:
            updated = expit(-(asymmetric_coef / deviation) * (residual - deviation))
        change = np.linalg.norm(updated - weights) / np.linalg.norm(weights)
        if change < tol:
            converged = True
            break
        weights = updated
        if model == "aspls":
            alpha = np.abs(residual) / np.max(np.abs(residual))
    restored = baseline * scale
    if not np.isfinite(restored).all():
        raise ValueError(f"{model} baseline produced non-finite values")
    return restored, converged
