# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
PSD (Phase-Sensitive Detection) signal-processing transform for spectroscopic data.

PSD is a deterministic signal-processing operator, conceptually analogous to
FFT or Hilbert transforms. It demodulates periodically modulated spectroscopic
data without learning any model parameters from data.
"""

import numpy as np
import traitlets as tr
from scipy.integrate import simpson

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.decorators import signature_has_configurable_traits

__all__ = ["PSD", "PSDResult"]
__configurables__ = ["PSD"]


# ======================================================================================
# PSDResult container
# ======================================================================================
class PSDResult:
    """
    Container for all outputs of a PSD transform.

    Attributes
    ----------
    prs : NDDataset
        Phase-resolved spectra with shape (n_phi, n_channels).
    in_phase : NDDataset
        In-phase component (phi=0°) with shape (n_channels,).
    quadrature : NDDataset
        Quadrature component (phi=90°) with shape (n_channels,).
    amplitude : NDDataset
        Amplitude = sqrt(in_phase² + quadrature²).
    phase : NDDataset
        Phase = atan2(quadrature, in_phase).
    T : NDDataset or None
        Transform matrix T with shape (n_phi, n_spectra_per_cycle).
        None when ``demodulation="integration"``.
    """

    def __init__(self, prs, in_phase, quadrature, amplitude, phase_lag, T=None):
        self.prs = prs
        self.in_phase = in_phase
        self.quadrature = quadrature
        self.amplitude = amplitude
        self.phase_lag = phase_lag
        self.T = T

    def __repr__(self):
        return (
            f"PSDResult(prs={self.prs.shape}, "
            f"in_phase={self.in_phase.shape}, "
            f"quadrature={self.quadrature.shape}, "
            f"amplitude={self.amplitude.shape}, "
            f"phase={self.phase_lag.shape}, "
            f"T={'None' if self.T is None else self.T.shape})"
        )


# ======================================================================================
# class PSD
# ======================================================================================
@signature_has_configurable_traits
class PSD(BaseConfigurable):
    """
    PSD (Phase-Sensitive Detection) for demodulating spectroscopic data.

    PSD is a deterministic signal-processing transform, analogous to FFT or
    Hilbert transforms. It does **not** learn from data and stores no model
    state.

    Supports two demodulation strategies:

    1. Matrix transform demodulation:
       ``A_demodulated = T · A_averaged``
    2. Explicit integration demodulation:
       ``A_demodulated(φ, λ) = (2/period) ∫ A_averaged(t, λ) · sin(k·ω·t + φ) dt``

    Parameters
    ----------
    demodulation : {"matrix", "integration"}, default="matrix"
        Demodulation strategy. Matrix demodulation is the default because it is generally faster for trapezoid and
        Simpson rules, while producing results equivalent to explicit integration.
    n_spectra_per_cycle : int or None, default=None
        Number of spectra per cycle. If None, inferred from data shape.
    harmonic : int, default=1
        Demodulation harmonic index (k in sin(k*ω*t + phi)).
    phi : array-like, default=np.arange(0.0, 360.0, 15.0)
        Phase angles for demodulation (in degrees).
    integration_rule : {"riemann", "trapezoid", "simpson"}, default="trapezoid"
        Integration rule for numerical integration weights.
        ``"riemann"`` uses a right-endpoint rectangular rule over n equal
        subintervals of [0, 1], sampling at ``t = (i+1)/n``. This rule
        gives exact results for pure sinusoids at any n.
    phase_unit : {"degrees", "radians"}, default="degrees"
        Unit for phase output.

    Notes
    -----
    - Requires phi to contain 0° and 90° for in_phase/quadrature extraction.
    - Matrix demodulation is default and faster.
    - Integration demodulation uses explicit numerical integration with a configurable integration rule.
    - The ``"riemann"`` rule uses a right-endpoint rectangular grid ``t = (i+1)/n``
      independent of the actual time coordinate; for irregular time coordinates
      use ``"trapezoid"`` or ``"simpson"``.
    - Constant offsets (DC) do not affect the PSD result because sinusoidal
      demodulation functions integrate to zero over one complete modulation
      period. No explicit mean subtraction is necessary.

    **Math notation:**

    - T = transform matrix (shape: n_phi × n_spectra_per_cycle)
    - period = modulation period (from time coordinate)
    - ω = 2π / period  (angular frequency)
    - harmonic = k (demodulation harmonic index)

    **PSD equation (matrix method):**

        A_demodulated = T · A_averaged

    where A_averaged has shape (n_spectra_per_cycle, n_channels),
    averaged across all cycles.

    **PSD equation (integration method):**

        A_demodulated (φ, λ) = (2/period) ∫ A_averaged(t, λ) · sin(k·ω·t_rel + φ) dt

    where φ is the demodulation phase angle, λ is the channel,
    and t_rel is normalized relative time within one modulation period
    (t_rel = 0 at the start of the period, t_rel = 1 at the end).

    Examples
    --------
    >>> import spectrochempy as scp
    >>> import numpy as np
    >>> # Raw 2D input (120 spectra, 1000 channels)
    >>> X = scp.NDDataset(np.random.rand(120, 1000))
    >>> psd = scp.PSD(n_spectra_per_cycle=60, demodulation='matrix')
    >>> result = psd.transform(X)
    >>> result.in_phase
    >>> result.quadrature
    >>> result.amplitude
    >>> result.phase
    """

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    demodulation = tr.Enum(
        ["matrix", "integration"],
        default_value="matrix",
    ).tag(config=True)

    n_spectra_per_cycle = tr.Integer(
        allow_none=True,
        default_value=None,
        help="Number of spectra per cycle.",
    ).tag(config=True)

    harmonic = tr.Integer(
        default_value=1,
        help="Demodulation harmonic index.",
    ).tag(config=True)

    phi = tr.Union(
        (tr.List(), Array()),
        help="Phase angles for demodulation (in degrees).",
    ).tag(config=True)

    integration_rule = tr.Enum(
        ["riemann", "trapezoid", "simpson"],
        default_value="trapezoid",
    ).tag(config=True)

    phase_unit = tr.Enum(
        ["degrees", "radians"],
        default_value="degrees",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        **kwargs,
    ):
        super().__init__(
            log_level=log_level,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Default trait values
    # ----------------------------------------------------------------------------------
    @tr.default("phi")
    def _default_phi(self):
        """Return a fresh array of default demodulation phase angles."""
        return np.arange(0.0, 360.0, 15.0)

    # ----------------------------------------------------------------------------------
    # Validation methods
    # ----------------------------------------------------------------------------------
    @tr.validate("harmonic")
    def _validate_harmonic(self, proposal):
        value = proposal.value
        if value <= 0:
            raise ValueError(f"harmonic must be positive. Got {value}.")
        return value

    @tr.validate("n_spectra_per_cycle")
    def _validate_n_spectra_per_cycle(self, proposal):
        value = proposal.value
        if value is not None and value <= 0:
            raise ValueError(f"n_spectra_per_cycle must be positive. Got {value}.")
        return value

    @tr.validate("phi")
    def _validate_phi(self, proposal):
        value = proposal.value
        phi_array = np.asarray(value)
        if phi_array.ndim != 1:
            raise ValueError(
                f"phi must be 1-dimensional. Got {phi_array.ndim} dimensions."
            )
        return value

    # ----------------------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------------------
    def _get_phi(self):
        """Return phi values as a float numpy array."""
        return np.asarray(self.phi, dtype=float)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _normalize_to_cycle_average(self, X):
        """
        Normalize input to the canonical 2D cycle-averaged modulation waveform.

        All supported input formats are reduced to a single cycle-averaged
        2D array ``A`` with shape ``(n_spectra_per_cycle, n_channels)``,
        which is the direct input to PSD demodulation.

        Input formats
        -------------
        - 3D grouped: ``(n_cycles, n_spectra_per_cycle, n_channels)``
        - 2D raw concatenated: ``(n_cycles * n_spectra_per_cycle, n_channels)``
          (requires ``n_spectra_per_cycle``)
        - 2D already averaged: ``(n_spectra_per_cycle, n_channels)``
          (when ``n_spectra_per_cycle`` is ``None``)

        Returns
        -------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra_per_cycle, n_channels).
        time_coord : Coord
            Time coordinate for the modulation axis.
        spectral_coord : Coord
            Spectral coordinate.
        n_spectra : int
            Number of spectra per cycle.
        n_channels : int
            Number of channels.
        """
        # Ensure X is NDDataset
        if not isinstance(X, NDDataset):
            X = NDDataset(X)

        ndim = X.ndim
        shape = X.shape
        n_spectra_per_cycle = self.n_spectra_per_cycle

        # Reject 1D and 4D+ input
        if ndim == 1:
            raise ValueError(
                f"1D input is not supported. Got shape {shape}. Use 2D or 3D input."
            )
        if ndim > 3:
            raise ValueError(
                f"{ndim}D input is not supported. Got shape {shape}. "
                "Use 2D or 3D input."
            )

        time_coord = None
        spectral_coord = None

        # Infer format from dimensionality and n_spectra_per_cycle
        if ndim == 3:
            # Grouped cycles: average over the cycle axis directly
            n_cycles, n_spectra, n_channels = shape
            A = np.mean(X.data, axis=0)

            if X.coordset is not None:
                dim_names = X.dims
                if len(dim_names) >= 3:
                    time_coord = X.coordset[dim_names[1]].copy()
                    spectral_coord = X.coordset[dim_names[2]].copy()

        elif ndim == 2:
            if n_spectra_per_cycle is None:
                # Already averaged / single cycle: return directly as 2D
                n_spectra, n_channels = shape
                A = X.data.copy()

                if X.coordset is not None:
                    dim_names = X.dims
                    if len(dim_names) >= 2:
                        time_coord = X.coordset[dim_names[0]].copy()
                        spectral_coord = X.coordset[dim_names[1]].copy()
            else:
                # Raw concatenated cycles: reshape, average, then flatten back
                if shape[0] % n_spectra_per_cycle != 0:
                    raise ValueError(
                        f"Input shape {shape} not divisible by n_spectra_per_cycle={n_spectra_per_cycle}. "
                        f"Expected first dimension to be N * {n_spectra_per_cycle}."
                    )
                n_cycles = shape[0] // n_spectra_per_cycle
                n_spectra = n_spectra_per_cycle
                n_channels = shape[1]
                data_3d = X.data.reshape(n_cycles, n_spectra_per_cycle, n_channels)
                A = np.mean(data_3d, axis=0)

                if X.coordset is not None:
                    dim_names = X.dims
                    if len(dim_names) >= 2:
                        # Raw concatenated: extract time coord from first dim
                        # and compute averaged relative time across cycles
                        full_time_coord = X.coordset[dim_names[0]]
                        full_time_data = np.asarray(full_time_coord.magnitude).reshape(
                            n_cycles, n_spectra_per_cycle
                        )
                        # Convert to relative time (subtract first time in each cycle)
                        times_rel = full_time_data - full_time_data[:, [0]]
                        # Average over cycles
                        ave_rel_times = np.mean(times_rel, axis=0)
                        # Create new time coordinate
                        time_coord = Coord(
                            ave_rel_times,
                            title=full_time_coord.title or "time",
                            units=full_time_coord.units,
                        )
                        spectral_coord = X.coordset[dim_names[1]].copy()

        # Create time coordinate if not available
        if time_coord is None:
            time_coord = Coord(
                np.linspace(0.0, 1.0, n_spectra),
                title="time",
                units=None,
            )

        # Create spectral coordinate if not available
        if spectral_coord is None:
            spectral_coord = Coord(
                np.arange(n_channels),
                title="channel",
                units=None,
            )

        return A, time_coord, spectral_coord, n_spectra, n_channels

    def _compute_T(self, n_spectra, time_coord):
        """
        Compute the transform matrix T.

        Parameters
        ----------
        n_spectra : int
            Number of spectra per cycle.
        time_coord : Coord
            Time coordinate for modulation axis.

        Returns
        -------
        NDDataset
            Transform matrix T with shape (n_phi, n_spectra_per_cycle).
        """
        n = n_spectra
        phi = self._get_phi()
        integration_rule = self.integration_rule.lower()
        harmonic = self.harmonic

        # Normalized time grid depends on the integration rule.
        # Riemann uses a right-endpoint rectangular rule over n equal subintervals
        # of [0, 1], sampling at t = (i+1)/n for i = 0..n-1.
        # Trapezoid and Simpson use the coordinate-derived grid spanning [0, 1].
        t_raw = np.asarray(time_coord.magnitude)

        if integration_rule == "riemann":
            # Right-endpoint rectangular rule: t = (i+1)/n for i = 0..n-1
            t_norm = (np.arange(n) + 1) / n
        elif len(t_raw) > 1:
            period = t_raw[-1] - t_raw[0]
            if period == 0:
                raise ValueError(
                    "Time coordinate has zero span (all values are identical). "
                    "Cannot perform PSD demodulation with a constant time coordinate."
                )
            t_norm = (t_raw - t_raw[0]) / period
        else:
            t_norm = np.array([0.0])

        # Integration weights.
        # For Riemann the grid is always uniform right-endpoint (independent of
        # actual time coordinate).  For trapezoid and Simpson the weights are
        # derived from the *actual* normalized coordinate spacing so that the
        # matrix result is algebraically identical to explicit numerical
        # integration (np.trapezoid / scipy.integrate.simpson) performed on the
        # same grid.
        if integration_rule == "riemann":
            w = np.ones(n)
            scaling = 2.0 / n
        elif integration_rule == "trapezoid":
            if n > 1:
                h = np.diff(t_norm)
                w = np.zeros(n)
                w[0] = 0.5 * h[0]
                if n > 2:
                    w[1:-1] = 0.5 * (h[:-1] + h[1:])
                w[-1] = 0.5 * h[-1]
            else:
                w = np.array([1.0])
            scaling = 2.0
        elif integration_rule == "simpson":
            if n < 3 or n % 2 == 0:
                raise ValueError(
                    f"Simpson's rule requires an odd number of points >= 3. Got n={n}. "
                    "Use an odd n_spectra_per_cycle >= 3 or choose a different integration rule."
                )
            h = np.diff(t_norm)
            w = np.zeros(n)
            for i in range(0, n - 2, 2):
                h0 = h[i]
                h1 = h[i + 1]
                hsum = h0 + h1
                hprod = h0 * h1
                w[i] += hsum / 6.0 * (2.0 - h1 / h0)
                w[i + 1] += hsum / 6.0 * hsum**2 / hprod
                w[i + 2] += hsum / 6.0 * (2.0 - h0 / h1)
            scaling = 2.0
        else:
            raise ValueError(f"Unknown integration rule: {integration_rule}")

        # Compute T matrix
        # T shape: (n_phi, n_spectra)
        phi_rad = phi[:, np.newaxis] * np.pi / 180.0
        t_norm_2d = t_norm[np.newaxis, :]

        T_data = scaling * w * np.sin(2.0 * np.pi * harmonic * t_norm_2d + phi_rad)

        # Create NDDataset
        T = NDDataset(T_data)
        T.dims = ["y", "x"]
        T.set_coordset(
            y=Coord(phi, title="demodulation phase angle", units="degrees"),
            x=time_coord.copy(),
        )
        T.title = "Demodulation coefficients"
        T.history = "Created by SpectroChemPy PSD"

        return T

    def _compute_psd_matrix(self, A, T_data):
        """
        Compute PSD using matrix method on cycle-averaged 2D data.

        Parameters
        ----------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra, n_channels).
        T_data : np.ndarray
            Transform matrix with shape (n_phi, n_spectra).

        Returns
        -------
        np.ndarray
            PSD result with shape (n_phi, n_channels).
        """
        # A shape: (n_spectra, n_channels)
        # T shape: (n_phi, n_spectra)
        # Result: (n_phi, n_channels)
        return np.dot(T_data, A)

    def _compute_psd_integration(self, A, n_spectra, time_coord_data):
        """
        Compute PSD using explicit integration on cycle-averaged 2D data.

        Parameters
        ----------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra, n_channels).
        n_spectra : int
            Number of spectra per cycle.
        time_coord_data : np.ndarray
            Time coordinate data.

        Returns
        -------
        np.ndarray
            Phase-resolved spectra (PRS) with shape (n_phi, n_channels).
        """
        n = n_spectra
        phi = self._get_phi()
        n_phi = len(phi)
        harmonic = self.harmonic
        integration_rule = self.integration_rule.lower()

        # Get time coordinate data
        time_data = time_coord_data
        if time_data is None:
            time_data = np.arange(n)

        # Period (must be non-zero)
        if n > 1:
            T_period = time_data[-1] - time_data[0]
            if T_period == 0:
                raise ValueError(
                    "Time coordinate has zero span (all values are identical). "
                    "Cannot perform PSD demodulation with a constant time coordinate."
                )
        else:
            T_period = 1.0

        # Normalized time grid depends on the integration rule.
        # Riemann uses a right-endpoint rectangular rule: t = (i+1)/n.
        # Trapezoid and Simpson use the coordinate-derived grid.
        if integration_rule == "riemann":
            t_norm = (np.arange(n) + 1) / n
        else:
            t_norm = (time_data - time_data[0]) / T_period

        # Convert phi to radians
        phi_rad = phi * np.pi / 180.0  # Shape: (n_phi,)

        channels = A.shape[1]

        # Result array: (n_phi, n_channels)
        psd = np.zeros((n_phi, channels))

        # angle: (n_phi, n_spectra)
        angle = (
            harmonic * 2.0 * np.pi * t_norm + phi_rad[:, np.newaxis]
        )  # (n_phi, n_spectra)
        # integrand: (n_phi, n_spectra, n_channels)
        integrand = A[np.newaxis, :, :] * np.sin(angle)[:, :, np.newaxis]

        # Integrate over spectra dimension (axis=1)
        if integration_rule == "simpson":
            psd = (2.0 / T_period) * simpson(integrand, time_data, axis=1)
        elif integration_rule == "riemann":
            # Right-endpoint rectangular rule: sum * (T_period / n)
            # Combined with global factor (2 / T_period) gives scaling 2/n.
            psd = (2.0 / T_period) * np.sum(integrand, axis=1) * (T_period / n)
        else:  # trapezoid
            psd = (2.0 / T_period) * np.trapezoid(integrand, time_data, axis=1)

        return psd

    def _extract_components(self, prs, phi):
        """
        Extract in_phase, quadrature, amplitude, and phase from prs.

        Parameters
        ----------
        prs : NDDataset
            Phase-resolved spectra with shape (n_phi, n_channels).
        phi : np.ndarray
            Phase angles.

        Returns
        -------
        tuple
            (in_phase, quadrature, amplitude, phase_lag) as NDDatasets.
        """
        # Find indices for 0° and 90° (with tolerance for floating point)
        idx_0 = np.where(np.isclose(phi, 0.0))[0]
        idx_90 = np.where(np.isclose(phi, 90.0))[0]

        if len(idx_0) == 0:
            raise ValueError(
                f"phi must contain 0° for in_phase extraction. phi values: {phi}"
            )
        if len(idx_90) == 0:
            raise ValueError(
                f"phi must contain 90° for quadrature extraction. phi values: {phi}"
            )

        idx_0 = idx_0[0]
        idx_90 = idx_90[0]

        # psd shape: (n_phi, n_channels)
        in_phase_data = prs.data[idx_0, :]
        quadrature_data = prs.data[idx_90, :]

        x_coord = prs.coordset["x"].copy()

        # Create NDDatasets
        in_phase = NDDataset(in_phase_data)
        in_phase.dims = ["x"]
        in_phase.set_coordset(x=x_coord)
        in_phase.units = prs.units
        in_phase.title = "In-phase spectrum (0°)"
        in_phase.history = "Created by SpectroChemPy PSD"

        quadrature = NDDataset(quadrature_data)
        quadrature.dims = ["x"]
        quadrature.set_coordset(x=x_coord)
        quadrature.units = prs.units
        quadrature.title = "Quadrature spectrum (90°)"
        quadrature.history = "Created by SpectroChemPy PSD"

        # Compute amplitude
        amplitude_data = np.sqrt(in_phase.data**2 + quadrature.data**2)
        amplitude = NDDataset(amplitude_data)
        amplitude.dims = in_phase.dims
        amplitude.set_coordset(**in_phase.coordset)
        amplitude.units = in_phase.units
        amplitude.title = "PSD Amplitude"
        amplitude.history = "Created by SpectroChemPy PSD"

        # Compute phase lag using np.arctan2
        phase_lag_data = np.arctan2(quadrature.data, in_phase.data)

        if self.phase_unit == "degrees":
            phase_lag_data = np.degrees(phase_lag_data)
            phase_lag_data = np.mod(phase_lag_data, 360.0)
            phase_lag_units = "degree"
        else:
            phase_lag_data = np.mod(phase_lag_data, 2.0 * np.pi)
            phase_lag_units = "radian"

        phase_lag = NDDataset(phase_lag_data)
        phase_lag.dims = in_phase.dims
        phase_lag.set_coordset(**in_phase.coordset)
        phase_lag.units = phase_lag_units
        phase_lag.title = "Phase lag"
        phase_lag.history = "Created by SpectroChemPy PSD"

        return in_phase, quadrature, amplitude, phase_lag

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def transform(self, X):
        """
        Apply the PSD transform to data X.

        All inputs are internally normalized into a single cycle-averaged
        2D modulation waveform ``A(t, ν)`` before demodulation.

        This is a pure computation that does not mutate internal state.
        Repeated calls are independent and thread-safe.

        Parameters
        ----------
        X : NDDataset or array-like
            Input data for PSD.

        Returns
        -------
        PSDResult
            Container with prs, in_phase, quadrature, amplitude, phase, and T.
        """
        # Normalize input to the canonical 2D cycle-averaged modulation waveform
        (
            A,
            time_coord,
            spectral_coord,
            n_spectra,
            n_channels,
        ) = self._normalize_to_cycle_average(X)

        # Compute PSD
        if self.demodulation == "matrix":
            T_matrix = self._compute_T(n_spectra, time_coord)
            T_data = T_matrix.data
            psd_data = self._compute_psd_matrix(A, T_data)
        else:
            T_matrix = None
            psd_data = self._compute_psd_integration(A, n_spectra, time_coord.magnitude)

        # Create NDDataset for psd
        # Shape: (n_phi, n_channels)
        psd = NDDataset(psd_data, dims=["y", "x"])
        y_coord = Coord(
            self._get_phi(), title=r"demodulation phase angle", units="degrees"
        )
        if spectral_coord is not None:
            x_coord = spectral_coord.copy()
        else:
            x_coord = Coord(
                np.arange(n_channels),
                title="channel",
                units=None,
            )
        psd.set_coordset(y=y_coord, x=x_coord)

        psd.units = X.units if hasattr(X, "units") else None
        psd.title = X.title if hasattr(X, "title") else "data"
        psd.history = "Created by SpectroChemPy PSD"

        # Extract in_phase (phi=0°), quadrature (phi=90°), amplitude, phase lag
        in_phase, quadrature, amplitude, phase_lag = self._extract_components(
            psd, self._get_phi()
        )

        return PSDResult(
            prs=psd,
            in_phase=in_phase,
            quadrature=quadrature,
            amplitude=amplitude,
            phase_lag=phase_lag,
            T=T_matrix,
        )

    def __call__(self, X):
        """Shorthand for ``self.transform(X)``."""
        return self.transform(X)

    def inverse_transform(self, X_transform=None, **kwargs):
        """
        Inverse transform is not supported for PSD.

        Raises
        ------
        NotImplementedError
            Always raised as PRS is not invertible.
        """
        raise NotImplementedError(
            "PRS is not invertible. inverse_transform() is not supported."
        )
