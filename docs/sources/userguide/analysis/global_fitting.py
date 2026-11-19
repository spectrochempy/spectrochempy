# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Fit spectral shapes and their evolution together
#
# A series can be described as a sum of separable components:
#
# $$X(t,x)=\sum_k A_k\,R(t;\theta_k)\,L(x;\phi_k).$$
#
# `Optimize(peaks=...)` adjusts every component jointly against the entire
# matrix. Its axes are identified by dataset dimension names, so transposing
# the matrix does not exchange the models. Here `x` is spectral frequency and
# `y` is recovery time. The same approach applies to other coordinate types.
#
# Each component has a coefficient `height` and a model on every axis.
# Parameters can vary independently, remain fixed, or reference a named
# shared parameter. A shared parameter contributes only one degree of freedom.

# %%
import spectrochempy as scp

frequency = scp.Coord.linspace(-8, 8, 101, units="Hz")
delay = scp.Coord([0, 0.3, 0.8, 1.6, 3, 6, 12, 30], units="s")
grid = scp.zeros((delay.size, frequency.size), coordset=[delay, frequency], units="V")
positions = [-5, -2, 1, 4]
truth = [
    {
        "height": height,
        "axes": {
            "x": {
                "model": "voigt",
                "parameters": {"center": center, "width": 1.5, "ratio": 0.7},
            },
            "y": {"model": "saturation_recovery", "parameters": {"tau": tau}},
        },
    }
    for height, center, tau in zip(
        [1, 2, 1.5, 0.8], positions, [1, 2, 4, 8], strict=True
    )
]
spectrum = scp.simulate_peaks(grid, truth)
spectrum += scp.normal(scale=0.003, size=grid.shape, seed=42, units=grid.units)
_ = spectrum.plot(method="lines", colorbar=True, title="Synthetic recovery series")

# %% [markdown]
# ## Choose spectral and recovery models
#
# Built-in spectral models are `gaussian`, `lorentzian`, `voigt` and
# `pseudovoigt`. Gaussian, Lorentzian and pseudo-Voigt use `center` and a
# positive FWHM `width`; pseudo-Voigt also uses `ratio`, the Gaussian fraction.
#
# Voigt uses Gaussian sigma `ratio * width / sqrt(8 * log(2))` and Lorentzian
# HWHM `(1-ratio) * width / 2`. Its `width` is a scale parameter, not the FWHM
# of the convolution. General Voigt profiles are normalized to one at their
# continuous center, independently of grid sampling. The existing 1D script
# models keep their existing conventions.
#
# `saturation_recovery` evaluates $1-\exp(-t/\tau)$;
# `exponential_decay` evaluates $\exp(-t/\tau)$. Both take positive `tau`.
# A freely fitted positive parameter needs a strictly positive lower bound.
# Bare numbers use the axis units; compatible quantities are converted.
#
# Here positions, widths, limiting amplitudes and recovery times are refined
# jointly. One Gaussian/Lorentzian ratio is shared by the four components.

# %%
peaks = [
    {
        "height": {"value": 1, "bounds": (0, 4)},
        "axes": {
            "x": {
                "model": "voigt",
                "parameters": {
                    "center": {
                        "value": center + 0.1,
                        "bounds": (center - 1, center + 1),
                    },
                    "width": {"value": 1.8, "bounds": (0.3, 4)},
                    "ratio": {"shared": "mixing"},
                },
            },
            "y": {
                "model": "saturation_recovery",
                "parameters": {
                    "tau": {"value": tau, "bounds": (0.1, 30)},
                },
            },
        },
    }
    for center, tau in zip(positions, [1.2, 2.5, 5, 10], strict=True)
]
fit = scp.Optimize(
    peaks=peaks,
    shared_parameters={
        "mixing": {"value": 0.5, "bounds": (0, 1)},
    },
).fit(spectrum)
result = fit.result
print("Converged:", result.diagnostics["success"])
print("Varying parameters:", result.diagnostics["n_varying_parameters"])
for component in result.peaks:
    print(
        "Center:",
        component["axes"]["x"]["parameters"]["center"],
        "Hz; tau:",
        component["axes"]["y"]["parameters"]["tau"],
        "s",
    )

# %% [markdown]
# ## Inspect the complete fit
#
# `result.components` contains one full matrix per component.
# `result.peaks` contains resolved fitted values that can be passed back to
# `simulate_peaks`. A coefficient is the limiting spectral height here because
# the recovery tends to one and the spectral profiles have unit height.
#
# The fit uses finite, unmasked measurements and ordinary unweighted least
# squares. Predictions retain the input coordinates and effective mask.
# Convergence and residuals should be inspected together. A good reconstruction
# alone does not establish that all component parameters are distinguishable.

# %%
_ = scp.plot_multiple(
    [spectrum[-1].squeeze(), result.fitted[-1].squeeze()],
    labels=["Observed", "Global fit"],
    method="pen",
    legend="best",
)
_ = result.residuals.plot(
    method="lines", colorbar=True, title="Residuals across all delays"
)
_ = scp.plot_multiple(
    [component[-1].squeeze() for component in result.components],
    labels=[f"Component {index + 1}" for index in range(4)],
    method="pen",
    legend="best",
    ylabel="Intensity / V",
    title="Fitted components at the longest delay",
)

# %% [markdown]
# Summing each fitted component over frequency gives its recovery profile.
# Normalize display copies to compare their shapes; the fitted amplitudes and
# reconstruction remain unchanged.

# %%
_ = scp.plot_multiple(
    [
        component.sum(dim="x").squeeze().normalize(dim=-1)
        for component in result.components
    ],
    labels=[f"Component {index + 1}" for index in range(4)],
    method="pen",
    legend="best",
    ylabel="Relative amplitude",
    title="Fitted recovery profiles",
)

# %% [markdown]
# ## Supply a law with several parameters
#
# A callable receives a copy of the numerical axis coordinate, in its displayed
# units, followed by keyword arguments from `parameters`. It must return a real,
# dimensionless vector of exactly the same length. `Optimize` does not normalize
# this vector: choose a convention that makes the coefficient interpretable.
#
# For example, a stretched recovery has a time constant and an exponent:


# %%
def stretched_recovery(t, tau, beta):
    return 1 - scp.exp(-((t / tau) ** beta))


stretched_axis = {
    "model": stretched_recovery,
    "parameters": {
        "tau": {"value": 2.0, "bounds": (0.1, 30), "unit": "s"},
        "beta": {"shared": "exponent"},
    },
}

# %% [markdown]
# Use this axis in each component and declare, for example,
# `shared_parameters={"exponent": {"value": 1, "bounds": (0.3, 1.5)}}`.
# Callback parameters are numerical magnitudes in their declared `unit`.
# Without a declaration, a quantity supplies its unit and an ordinary scalar
# is dimensionless. A shared reference without a unit uses the shared
# definition's unit. To convert a shared time defined in milliseconds for a
# callback receiving seconds, use `{"shared": "time", "unit": "s"}`.
#
# Sharing also works for coefficients and parameters across different axes.
# Unknown references, incompatible units and unused shared definitions are
# rejected. Fixed shared parameters use `fixed=True` and are omitted from the
# varying parameter count. The old flat Gaussian/Lorentzian syntax remains valid.
#
# ## Connect an external spectral simulator
#
# A spectral callback can compute a quadrupolar profile using the optional
# NMR plugin's PULSAR simulator. Keep its sampling, broadening and powder
# settings fixed during the fit. The callback must evaluate or interpolate
# the simulated spectrum onto the coordinate it receives. For an Hz axis:
#
# ```python
# pulsar = scp.nmr.pulsar
#
# def quadrupolar(frequency, cq_hz, eta):
#     powder = pulsar.simulate_central_transition(
#         cq_hz=cq_hz, eta=eta, larmor_hz=104e6,
#         spectral_width_hz=20000, points=2048, accuracy=16,
#     )
#     step = powder.x.data[1] - powder.x.data[0]
#     powder = pulsar.broaden(powder, gaussian_fwhm_hz=200)
#     powder.set_coordset(x=powder.x + (step / 2) * scp.ur.Hz)
#     sampled = powder.interpolate(coord=scp.Coord(frequency, units="Hz"))
#     return sampled.data / float(powder.max())
#
# quadrupolar_axis = {
#     "model": quadrupolar,
#     "parameters": {
#         "cq_hz": {"value": 2e6, "bounds": (1e6, 3e6), "unit": "Hz"},
#         "eta": {"value": 0.3, "bounds": (0, 1)},
#     },
# }
# ```
#
# This adapter normalizes to the maximum on the simulator grid. Its coefficient
# follows that convention. PULSAR labels bin left edges; the adapter shifts to
# bin centers. The simulator grid must cover the fitted domain.
# Fit quality and parameter stability should be checked as numerical accuracy
# and broadening are varied. Callbacks use numerical derivatives; binned or
# nonsmooth simulations may require smoother evaluation before fitting.
#
# ## Read parameter diagnostics
#
# `result.parameter_names` and `result.parameter_units` describe the columns of
# `fit.jacobian`. Each shared parameter has one column that includes all of its
# contributions. Local covariance, standard errors and confidence intervals
# follow the existing ND fitting contract. They are unavailable when the fit
# fails, lacks degrees of freedom, is rank deficient or has an active bound;
# inspect `result.diagnostics["covariance_reason"]` in that case.
#
# Fitted datasets retain numerical parameters and callable identities in their
# history. They do not serialize executable callback code. Keep callback source
# and external simulator settings with the workflow for reproducibility.
