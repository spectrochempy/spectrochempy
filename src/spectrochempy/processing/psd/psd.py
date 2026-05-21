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
        Phase-resolved spectra with shape (n_phi, n_wavenumbers).
    in_phase : NDDataset
        In-phase component (phi=0°) with shape (n_wavenumbers,).
    quadrature : NDDataset
        Quadrature component (phi=90°) with shape (n_wavenumbers,).
    amplitude : NDDataset
        Amplitude = sqrt(in_phase² + quadrature²).
    phase : NDDataset
        Phase = atan2(quadrature, in_phase).
    T : NDDataset or None
        Transform matrix T with shape (n_phi, n_spectra_per_cycle).
        None when ``demodulation="integration"``.
    """

    def __init__(self, prs, in_phase, quadrature, amplitude, phase, T=None):
        self.prs = prs
        self.in_phase = in_phase
        self.quadrature = quadrature
        self.amplitude = amplitude
        self.phase = phase
        self.T = T

    def __repr__(self):
        return (
            f"PSDResult(prs={self.prs.shape}, "
            f"in_phase={self.in_phase.shape}, "
            f"quadrature={self.quadrature.shape}, "
            f"amplitude={self.amplitude.shape}, "
            f"phase={self.phase.shape}, "
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
    phase_unit : {"degrees", "radians"}, default="degrees"
        Unit for phase output.

    Notes
    -----
    - Requires phi to contain 0° and 90° for in_phase/quadrature extraction.
    - Matrix demodulation is default and faster.
    - Integration demodulation uses explicit numerical integration with a configurable integration rule.
    - The ``"riemann"`` rule assumes uniform sampling (equal time steps); for
      irregular time coordinates use ``"trapezoid"`` or ``"simpson"``.
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

    where A_averaged has shape (n_spectra_per_cycle, n_wavenumbers),
    averaged across all cycles.

    **PSD equation (integration method):**

        A_demodulated (φ, λ) = (2/period) ∫ A_averaged(t, λ) · sin(k·ω·t_rel + φ) dt

    where φ is the demodulation phase angle, λ is wavenumber,
    and t_rel is normalized relative time within one modulation period
    (t_rel = 0 at the start of the period, t_rel = 1 at the end).

    Examples
    --------
    >>> import spectrochempy as scp
    >>> import numpy as np
    >>> # Raw 2D input (120 spectra, 1000 wavenumbers)
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
        default_value=None,
        help="Phase angles for demodulation (in degrees).",
    ).tag(config=True)

    # Default phi values as tuple (to avoid array comparison issues)
    _default_phi = (
        0.0,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
        90.0,
        105.0,
        120.0,
        135.0,
        150.0,
        165.0,
        180.0,
        195.0,
        210.0,
        225.0,
        240.0,
        255.0,
        270.0,
        285.0,
        300.0,
        315.0,
        330.0,
        345.0,
    )

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
        if value is None:
            return None
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
        """Get phi values, using default if not set."""
        if self.phi is None:
            return np.array(self._default_phi)
        return np.asarray(self.phi)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _normalize_to_cycle_average(self, X):
        """
        Normalize input to the canonical 2D cycle-averaged modulation waveform.

        All supported input formats are reduced to a single cycle-averaged
        2D array ``A`` with shape ``(n_spectra_per_cycle, n_wavenumbers)``,
        which is the direct input to PSD demodulation.

        Input formats
        -------------
        - 3D grouped: ``(n_cycles, n_spectra_per_cycle, n_wavenumbers)``
        - 2D raw concatenated: ``(n_cycles * n_spectra_per_cycle, n_wavenumbers)``
          (requires ``n_spectra_per_cycle``)
        - 2D already averaged: ``(n_spectra_per_cycle, n_wavenumbers)``
          (when ``n_spectra_per_cycle`` is ``None``)

        Returns
        -------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra_per_cycle, n_wavenumbers).
        time_coord : Coord
            Time coordinate for the modulation axis.
        spectral_coord : Coord
            Spectral coordinate.
        n_spectra : int
            Number of spectra per cycle.
        n_wavenumbers : int
            Number of wavenumbers.
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
            n_cycles, n_spectra, n_wavenumbers = shape
            A = np.mean(X.data, axis=0)

            if X.coordset is not None:
                dim_names = X.dims
                if len(dim_names) >= 3:
                    time_coord = X.coordset[dim_names[1]].copy()
                    spectral_coord = X.coordset[dim_names[2]].copy()

        elif ndim == 2:
            if n_spectra_per_cycle is None:
                # Already averaged / single cycle: return directly as 2D
                n_spectra, n_wavenumbers = shape
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
                n_wavenumbers = shape[1]
                data_3d = X.data.reshape(n_cycles, n_spectra_per_cycle, n_wavenumbers)
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
                np.arange(n_wavenumbers),
                title="wavenumber",
                units=None,
            )

        return A, time_coord, spectral_coord, n_spectra, n_wavenumbers

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

        # Normalized time from 0 to 1
        # Use the raw time values (magnitude) so that matrix and integration
        # methods share the exact same grid and agree numerically.
        t_raw = np.asarray(time_coord.magnitude)
        if len(t_raw) > 1:
            period = t_raw[-1] - t_raw[0]
            if period == 0:
                raise ValueError(
                    "Time coordinate has zero span (all values are identical). "
                    "Cannot perform PSD demodulation with a constant time coordinate."
                )
            t_norm = (t_raw - t_raw[0]) / period
        else:
            t_norm = np.array([0.0])

        # Integration weights
        if integration_rule == "riemann":
            w = np.ones(n)
        elif integration_rule == "trapezoid":
            w = np.ones(n)
            if n > 1:
                w[0] = 0.5
                w[-1] = 0.5
        elif integration_rule == "simpson":
            if n < 3 or n % 2 == 0:
                raise ValueError(
                    f"Simpson's rule requires an odd number of points >= 3. Got n={n}. "
                    "Use an odd n_spectra_per_cycle >= 3 or choose a different integration rule."
                )
            w = np.ones(n)
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
        else:
            raise ValueError(f"Unknown integration rule: {integration_rule}")

        # Compute T matrix
        # T shape: (n_phi, n_spectra)
        phi_rad = phi[:, np.newaxis] * np.pi / 180.0
        t_norm_2d = t_norm[np.newaxis, :]

        # Scaling incorporates the integration step size and the global factor 2
        # (from PSD(phi) = 2 * integral_0^1 D(t) sin(...) dt).
        if integration_rule == "trapezoid":
            # h = 1/(n-1);  scaling = 2 * h = 2/(n-1)
            scaling = 2.0 / (n - 1) if n > 1 else 2.0
        elif integration_rule == "simpson":
            # h = 1/(n-1);  scaling = 2 * h / 3 = 2/(3*(n-1))
            scaling = 2.0 / (3.0 * (n - 1))
        else:
            # riemann: h = 1/n;  scaling = 2 * h = 2/n
            scaling = 2.0 / n

        T_data = scaling * w * np.sin(2.0 * np.pi * harmonic * t_norm_2d + phi_rad)

        # Create NDDataset
        T = NDDataset(T_data)
        T.dims = ["y", "x"]
        T.set_coordset(
            y=Coord(phi, title="demodulation phase angle", units="degrees"),
            x=time_coord.copy(),
        )
        T.title = "PSD transform matrix T"
        T.history = "Created by SpectroChemPy PSD"

        return T

    def _compute_psd_matrix(self, A, T_data):
        """
        Compute PSD using matrix method on cycle-averaged 2D data.

        Parameters
        ----------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra, n_wavenumbers).
        T_data : np.ndarray
            Transform matrix with shape (n_phi, n_spectra).

        Returns
        -------
        np.ndarray
            PSD result with shape (n_phi, n_wavenumbers).
        """
        # A shape: (n_spectra, n_wavenumbers)
        # T shape: (n_phi, n_spectra)
        # Result: (n_phi, n_wavenumbers)
        return np.dot(T_data, A)

    def _compute_psd_integration(self, A, n_spectra, time_coord_data):
        """
        Compute PSD using explicit integration on cycle-averaged 2D data.

        Parameters
        ----------
        A : np.ndarray
            Cycle-averaged data with shape (n_spectra, n_wavenumbers).
        n_spectra : int
            Number of spectra per cycle.
        time_coord_data : np.ndarray
            Time coordinate data.

        Returns
        -------
        np.ndarray
            Phase-resolved spectra (PRS) with shape (n_phi, n_wavenumbers).
        """
        n = n_spectra
        phi = self._get_phi()
        n_phi = len(phi)
        harmonic = self.harmonic

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

        # Normalized relative time: t_rel = 0 at start, t_rel = 1 at end of period.
        # This ensures matrix and integration methods agree regardless of absolute origin.
        t_rel = (time_data - time_data[0]) / T_period

        # Convert phi to radians
        phi_rad = phi * np.pi / 180.0  # Shape: (n_phi,)

        n_wavenumbers = A.shape[1]

        # Result array: (n_phi, n_wavenumbers)
        psd = np.zeros((n_phi, n_wavenumbers))

        # angle: (n_phi, n_spectra)
        angle = (
            harmonic * 2.0 * np.pi * t_rel + phi_rad[:, np.newaxis]
        )  # (n_phi, n_spectra)
        # integrand: (n_phi, n_spectra, n_wavenumbers)
        integrand = A[np.newaxis, :, :] * np.sin(angle)[:, :, np.newaxis]

        # Integrate over spectra dimension (axis=1)
        integration_rule = self.integration_rule.lower()
        if integration_rule == "simpson":
            psd = (2.0 / T_period) * simpson(integrand, time_data, axis=1)
        elif integration_rule == "riemann":
            # Uniform rectangular rule: step = T_period / n
            # Matches the matrix-method scaling of 2/n.
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
            Phase-resolved spectra with shape (n_phi, n_wavenumbers).
        phi : np.ndarray
            Phase angles.

        Returns
        -------
        tuple
            (in_phase, quadrature, amplitude, phase) as NDDatasets.
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

        # psd shape: (n_phi, n_wavenumbers)
        in_phase_data = prs.data[idx_0, :]
        quadrature_data = prs.data[idx_90, :]

        x_coord = prs.coordset["x"].copy()

        # Create NDDatasets
        in_phase = NDDataset(in_phase_data)
        in_phase.dims = ["x"]
        in_phase.set_coordset(x=x_coord)
        in_phase.units = prs.units
        in_phase.title = "In-phase component (0°)"
        in_phase.history = "Created by SpectroChemPy PSD"

        quadrature = NDDataset(quadrature_data)
        quadrature.dims = ["x"]
        quadrature.set_coordset(x=x_coord)
        quadrature.units = prs.units
        quadrature.title = "Quadrature component (90°)"
        quadrature.history = "Created by SpectroChemPy PSD"

        # Compute amplitude
        amplitude_data = np.sqrt(in_phase.data**2 + quadrature.data**2)
        amplitude = NDDataset(amplitude_data)
        amplitude.dims = in_phase.dims
        amplitude.set_coordset(**in_phase.coordset)
        amplitude.units = in_phase.units
        amplitude.title = "PSD Amplitude"
        amplitude.history = "Created by SpectroChemPy PSD"

        # Compute phase using np.arctan2
        phase_data = np.arctan2(quadrature.data, in_phase.data)
        if self.phase_unit == "degrees":
            phase_data = phase_data * 180.0 / np.pi
        phase = NDDataset(phase_data)
        phase.dims = in_phase.dims
        phase.set_coordset(**in_phase.coordset)
        phase.units = "degree" if self.phase_unit == "degrees" else "radian"
        phase.title = "PSD Phase"
        phase.history = "Created by SpectroChemPy PSD"

        return in_phase, quadrature, amplitude, phase

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
            n_wavenumbers,
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
        # Shape: (n_phi, n_wavenumbers)
        psd = NDDataset(psd_data, dims=["y", "x"])
        y_coord = Coord(self._get_phi(), title="demodulation phase", units="degrees")
        if spectral_coord is not None:
            x_coord = spectral_coord.copy()
        else:
            x_coord = Coord(
                np.arange(n_wavenumbers),
                title="wavenumber",
                units=None,
            )
        psd.set_coordset(y=y_coord, x=x_coord)

        psd.units = X.units if hasattr(X, "units") else None
        psd.title = f"PSD of {X.title if hasattr(X, 'title') else 'data'}"
        psd.history = "Created by SpectroChemPy PSD"

        # Extract in_phase (phi=0°), quadrature (phi=90°), amplitude, phase
        in_phase, quadrature, amplitude, phase = self._extract_components(
            psd, self._get_phi()
        )

        return PSDResult(
            prs=psd,
            in_phase=in_phase,
            quadrature=quadrature,
            amplitude=amplitude,
            phase=phase,
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
