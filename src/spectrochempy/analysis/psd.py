# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Implementation of PSD (Phase-Sensitive Detection) for spectroscopic data."""

import numpy as np
import traitlets as tr
from scipy.integrate import simpson

from spectrochempy.analysis._base._analysisbase import AnalysisConfigurable
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits

__all__ = ["PSD"]
__configurables__ = ["PSD"]


# ======================================================================================
# class PSD
# ======================================================================================
@signature_has_configurable_traits
class PSD(AnalysisConfigurable):
    """
    PSD (Phase-Sensitive Detection) for demodulating spectroscopic data.

    Supports two computation methods:
    1. Matrix transform method: demodulated = np.dot(T, A)
    2. Explicit integration method: PSD(phi, nu) = 2/T * ∫ D(t,nu) * sin(k*ω*t + phi) dt

    Parameters
    ----------
    method : {"matrix", "integration"}, default="matrix"
        Computation method. "matrix" is faster using linear algebra.
    n_spectra_per_cycle : int or None, default=None
        Number of spectra per cycle. If None, inferred from data shape.
    input_mode : {"auto", "raw", "grouped", "averaged"}, default="auto"
        Input data format:
        - "raw": 2D (N_cycles*n_spectra_per_cycle, n_wavenumbers)
        - "grouped": 3D (N_cycles, n_spectra_per_cycle, n_wavenumbers)
        - "averaged": 2D (n_spectra_per_cycle, n_wavenumbers)
        - "auto": infer from shape
    harmonic : int, default=1
        Demodulation harmonic index (k in sin(k*ω*t + phi)).
    phi : array-like, default=np.arange(0.0, 360.0, 15.0)
        Phase angles for demodulation (in degrees).
    integration_method : {"rectangle", "riemann", "trapezoid", "simpson"}, default="trapezoid"
        Quadrature method for numerical integration weights.
    subtract_mean : bool, default=False
        If True, subtract mean from data before demodulation.
    phase_unit : {"degrees", "radians"}, default="degrees"
        Unit for phase output.

    Attributes
    ----------
    T : NDDataset
        Transform matrix (NOT the period). Shape: (n_phi, n_spectra_per_cycle).
        Dims: ["y", "x"] where y=phi angles, x=time/spectra.
    period : float
        Modulation period (from time coordinate).
    prs : NDDataset
        Phase-resolved spectra (PRS) with shape (n_phi, n_wavenumbers) and dims ["y", "x"].
    in_phase : NDDataset
        In-phase component (phi=0°) with shape (n_wavenumbers,).
    quadrature : NDDataset
        Quadrature component (phi=90°) with shape (n_wavenumbers,).
    amplitude : NDDataset
        Amplitude = sqrt(in_phase² + quadrature²) with shape (n_wavenumbers,).
    phase : NDDataset
        Phase = atan2(quadrature, in_phase) with shape (n_wavenumbers,).

    Methods
    -------
    fit(X)
        Fit the PSD model to data X.
    transform(X)
        Transform data using fitted model (returns PSD).
    inverse_transform()
        Not supported for PSD.

    Notes
    -----
    - Requires phi to contain 0° and 90° for in_phase/quadrature extraction.
    - Matrix method is default and faster.
    - Integration method uses explicit numerical integration with configurable quadrature.

    **Math notation:**

    - T = transform matrix (shape: n_phi × n_spectra_per_cycle)
    - period = modulation period (from time coordinate)
    - ω = 2π / period  (angular frequency)
    - harmonic = k (demodulation harmonic index)

    **PSD equation (matrix method):**

        A = T · D_averaged

    where D_averaged has shape (n_spectra_per_cycle, n_wavenumbers),
    averaged across all cycles.

    **PSD equation (integration method):**

        PSD(φ, λ) = (2/period) ∫ D_averaged(t, λ) · sin(k·ω·t + φ) dt

    where φ is the demodulation phase angle, λ is wavenumber,
    and t is normalized time within one modulation period.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> import numpy as np
    >>> # Raw 2D input (120 spectra, 1000 wavenumbers)
    >>> X = scp.NDDataset(np.random.rand(120, 1000))
    >>> psd = scp.PSD(n_spectra_per_cycle=60, method='matrix')
    >>> psd.fit(X)
    >>> in_phase = psd.in_phase
    >>> quadrature = psd.quadrature
    >>> amplitude = psd.amplitude
    >>> phase = psd.phase
    """

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    method = tr.Enum(
        ["matrix", "integration"],
        default_value="matrix",
    ).tag(config=True)

    n_spectra_per_cycle = tr.Integer(
        allow_none=True,
        default_value=None,
        help="Number of spectra per cycle.",
    ).tag(config=True)

    input_mode = tr.Enum(
        ["auto", "raw", "grouped", "averaged"],
        default_value="auto",
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

    integration_method = tr.Enum(
        ["rectangle", "riemann", "trapezoid", "simpson"],
        default_value="trapezoid",
    ).tag(config=True)

    subtract_mean = tr.Bool(
        default_value=False,
        help="If True, subtract mean from data before demodulation.",
    ).tag(config=True)

    phase_unit = tr.Enum(
        ["degrees", "radians"],
        default_value="degrees",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _T = tr.Instance(NDDataset, allow_none=True, help="Transform matrix.")
    _prs = tr.Instance(NDDataset, allow_none=True, help="Phase-resolved spectra.")
    _in_phase = tr.Instance(NDDataset, allow_none=True, help="In-phase component.")
    _quadrature = tr.Instance(NDDataset, allow_none=True, help="Quadrature component.")
    _amplitude = tr.Instance(NDDataset, allow_none=True, help="Amplitude.")
    _phase = tr.Instance(NDDataset, allow_none=True, help="Phase.")

    _normalized_data = Array(
        help="3D normalized data (N_cycles, n_spectra, n_wavenumbers)."
    )
    _n_cycles = tr.Integer(allow_none=True)
    _n_spectra = tr.Integer(allow_none=True)
    _n_wavenumbers = tr.Integer(allow_none=True)
    _time_coord = tr.Instance(
        Coord, allow_none=True, help="Time coordinate for modulation."
    )
    _cycle_coord = tr.Instance(Coord, allow_none=True, help="Cycle coordinate.")
    _spectral_coord = tr.Instance(Coord, allow_none=True, help="Spectral coordinate.")

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
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
    def _normalize_input(self, X):
        """
        Normalize input to internal 3D representation.

        Returns
        -------
        data_3d : np.ndarray
            3D array of shape (N_cycles, n_spectra_per_cycle, n_wavenumbers)
        time_coord : Coord or None
            Time coordinate for modulation axis
        cycle_coord : Coord or None
            Cycle coordinate
        spectral_coord : Coord or None
            Spectral coordinate
        """
        # Ensure X is NDDataset
        if not isinstance(X, NDDataset):
            X = NDDataset(X)

        ndim = X.ndim
        shape = X.shape

        # Reject 1D and 4D+ input
        if ndim == 1:
            raise ValueError(
                f"1D input is not supported. Got shape {shape}. " "Use 2D or 3D input."
            )
        if ndim > 3:
            raise ValueError(
                f"{ndim}D input is not supported. Got shape {shape}. "
                "Use 2D or 3D input."
            )

        # Determine input mode
        input_mode = self.input_mode
        n_spectra_per_cycle = self.n_spectra_per_cycle

        if input_mode == "auto":
            if ndim == 3:
                input_mode = "grouped"
            elif ndim == 2:
                if n_spectra_per_cycle is not None and shape[0] == n_spectra_per_cycle:
                    input_mode = "averaged"
                elif (
                    n_spectra_per_cycle is not None
                    and shape[0] % n_spectra_per_cycle == 0
                ):
                    input_mode = "raw"
                else:
                    raise ValueError(
                        f"Cannot infer input_mode for 2D input with shape {shape}. "
                        "Please specify input_mode and/or n_spectra_per_cycle."
                    )

        # Process based on input mode
        if input_mode == "grouped":
            if ndim != 3:
                raise ValueError(
                    f"input_mode='grouped' requires 3D input. Got {ndim}D with shape {shape}."
                )
            data_3d = X.data.copy()
            n_cycles, n_spectra, n_wavenumbers = shape

        elif input_mode == "raw":
            if ndim != 2:
                raise ValueError(
                    f"input_mode='raw' requires 2D input. Got {ndim}D with shape {shape}."
                )
            if n_spectra_per_cycle is None:
                raise ValueError(
                    "n_spectra_per_cycle must be specified for input_mode='raw'."
                )
            if shape[0] % n_spectra_per_cycle != 0:
                raise ValueError(
                    f"Input shape {shape} not divisible by n_spectra_per_cycle={n_spectra_per_cycle}. "
                    f"Expected first dimension to be N * {n_spectra_per_cycle}."
                )
            n_cycles = shape[0] // n_spectra_per_cycle
            n_wavenumbers = shape[1]
            data_3d = X.data.copy().reshape(
                n_cycles, n_spectra_per_cycle, n_wavenumbers
            )
            n_spectra = n_spectra_per_cycle

        elif input_mode == "averaged":
            if ndim != 2:
                raise ValueError(
                    f"input_mode='averaged' requires 2D input. Got {ndim}D with shape {shape}."
                )
            if n_spectra_per_cycle is None:
                raise ValueError(
                    "n_spectra_per_cycle must be specified for input_mode='averaged'."
                )
            if shape[0] != n_spectra_per_cycle:
                raise ValueError(
                    f"For input_mode='averaged', shape[0] must equal n_spectra_per_cycle. "
                    f"Got shape[0]={shape[0]}, n_spectra_per_cycle={n_spectra_per_cycle}."
                )
            n_cycles = 1
            n_spectra = n_spectra_per_cycle
            n_wavenumbers = shape[1]
            data_3d = X.data.copy()[
                np.newaxis, :, :
            ]  # Shape: (1, n_spectra, n_wavenumbers)

        # Get coordinates
        time_coord = None
        cycle_coord = None
        spectral_coord = None

        if X.coordset is not None:
            # For 3D grouped input
            if input_mode == "grouped" and ndim == 3:
                # Assume dims are (cycle, time, spectral)
                dim_names = X.dims
                if len(dim_names) >= 3:
                    cycle_coord = X.coordset[dim_names[0]].copy()
                    time_coord = X.coordset[dim_names[1]].copy()
                    spectral_coord = X.coordset[dim_names[2]].copy()
            # For 2D input
            elif ndim == 2:
                dim_names = X.dims
                if len(dim_names) >= 2:
                    if input_mode == "raw":
                        # For raw input, extract time coord from first dim
                        # and compute averaged relative time across cycles
                        full_time_coord = X.coordset[dim_names[0]]
                        # Reshape to (n_cycles, n_spectra_per_cycle)
                        full_time_data = full_time_coord.data.reshape(
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
                    else:  # averaged
                        time_coord = X.coordset[dim_names[0]].copy()
                        spectral_coord = X.coordset[dim_names[1]].copy()

        # Create time coordinate if not available
        if time_coord is None:
            # Prefer linspace(0, 1, n_spectra) to include both 0 and 1
            time_coord = Coord(
                np.linspace(0.0, 1.0, n_spectra),
                title="time",
                units=None,
            )

        # Create cycle coordinate if not available
        if cycle_coord is None:
            cycle_coord = Coord(
                np.arange(n_cycles),
                title="cycle",
                units=None,
            )

        # Create spectral coordinate if not available
        if spectral_coord is None:
            spectral_coord = Coord(
                np.arange(n_wavenumbers),
                title="wavenumber",
                units=None,
            )

        # Store dimensions
        self._n_cycles = n_cycles
        self._n_spectra = n_spectra
        self._n_wavenumbers = n_wavenumbers
        self._time_coord = time_coord
        self._cycle_coord = cycle_coord

        return data_3d, time_coord, cycle_coord, spectral_coord

    def _compute_T(self, time_coord):
        """
        Compute the transform matrix T.

        Parameters
        ----------
        time_coord : Coord
            Time coordinate for modulation axis.

        Returns
        -------
        NDDataset
            Transform matrix T with shape (n_phi, n_spectra_per_cycle).
        """
        n = self._n_spectra
        phi = self._get_phi()
        quadrature = self.integration_method.lower()
        harmonic = self.harmonic

        # Normalized time from 0 to 1
        # For trapezoid: use (arange(n)) / (n-1)
        # For rectangle/riemann and simpson: use (arange(n) + 1) / n
        if quadrature in {"trapezoid"}:
            t_norm = np.arange(n) / (n - 1) if n > 1 else np.array([0.0])
        else:
            t_norm = (np.arange(n) + 1) / n

        # Quadrature weights
        if quadrature in {"rectangle", "riemann"}:
            w = np.ones(n)
        elif quadrature == "trapezoid":
            w = np.ones(n)
            if n > 1:
                w[0] = 0.5
                w[-1] = 0.5
        elif quadrature == "simpson":
            if n % 2 == 0:
                raise ValueError(
                    f"Simpson's rule requires an odd number of points. Got n={n}. "
                    "Use an odd n_spectra_per_cycle or choose a different quadrature method."
                )
            w = np.ones(n)
            w[0::2] = 4.0 / 3.0
            w[1::2] = 2.0 / 3.0
        else:
            raise ValueError(f"Unknown quadrature method: {quadrature}")

        # Compute T matrix
        # T shape: (n_phi, n_spectra)
        phi_rad = phi[:, np.newaxis] * np.pi / 180.0
        t_norm_2d = t_norm[np.newaxis, :]

        # Scaling: 2/(n-1) for trapezoid (matches integral approximation)
        # For other methods, 2/n is correct
        if quadrature == "trapezoid":
            scaling = 2.0 / (n - 1) if n > 1 else 2.0
        else:
            scaling = 2.0 / n

        T_data = scaling * w * np.sin(2.0 * np.pi * harmonic * t_norm_2d + phi_rad)

        # Create NDDataset
        T = NDDataset(T_data)
        T.dims = ["y", "x"]
        T.set_coordset(
            y=Coord(phi, title="y", units="degrees"),
            x=time_coord.copy(),
        )
        T.title = "PSD transform matrix T"
        T.history = "Created by SpectroChemPy PSD"

        return T

    def _compute_psd_matrix(self, A):
        """
        Compute PSD using matrix method.

        Parameters
        ----------
        A : np.ndarray
            Data with shape (n_spectra, n_wavenumbers) or (N_cycles, n_spectra, n_wavenumbers).

        Returns
        -------
        np.ndarray
            PSD result.
        """
        T_data = self._T.data  # Shape: (n_phi, n_spectra)

        # Per-cycle computation: apply T to each cycle independently
        # A shape: (N_cycles, n_spectra, n_wavenumbers)
        # T shape: (n_phi, n_spectra)
        # Result: (N_cycles, n_phi, n_wavenumbers)
        # Use einsum: T[phi, spec] * A[cycle, spec, wn] -> [cycle, phi, wn]
        if A.ndim == 3:  # noqa: SIM108
            psd = np.einsum("ps,csi->cpi", T_data, A)
        else:
            # Averaged or single cycle: A shape (n_spectra, n_wavenumbers)
            # Result: (n_phi, n_wavenumbers)
            psd = np.dot(T_data, A)

        return psd

    def _compute_psd_integration(self, D):
        """
        Compute PSD using explicit integration method (vectorized).

        Parameters
        ----------
        D : np.ndarray
            Data with shape (n_spectra, n_wavenumbers) or (N_cycles, n_spectra, n_wavenumbers).

        Returns
        -------
        np.ndarray
            Phase-resolved spectra (PRS) result.
        """
        n = self._n_spectra
        phi = self._get_phi()
        n_phi = len(phi)
        harmonic = self.harmonic

        # Get time coordinate data
        time_data = self._time_coord.data
        if time_data is None:
            time_data = np.arange(n)

        # Period
        T_period = time_data[-1] - time_data[0] if n > 1 else 1.0

        # Convert phi to radians
        phi_rad = phi * np.pi / 180.0  # Shape: (n_phi,)

        if D.ndim == 3:
            # Per-cycle computation: D shape (N_cycles, n_spectra, n_wavenumbers)
            N_cycles = D.shape[0]
            n_wavenumbers = D.shape[2]

            # Result array: (N_cycles, n_phi, n_wavenumbers)
            psd = np.zeros((N_cycles, n_phi, n_wavenumbers))

            for i in range(N_cycles):
                # Compute integrand for all phi at once
                # D[i]: (n_spectra, n_wavenumbers)
                # angle: (n_phi, n_spectra)
                angle = (
                    harmonic * 2.0 * np.pi * time_data / T_period
                    + phi_rad[:, np.newaxis]
                )  # (n_phi, n_spectra)
                # integrand: (n_phi, n_spectra, n_wavenumbers)
                integrand = D[i][np.newaxis, :, :] * np.sin(angle)[:, :, np.newaxis]

                # Integrate over spectra dimension (axis=1)
                # The integral is: ∫ D * sin(...) dt from 0 to T
                # Using normalized time t' = t/T, dt = T * dt'
                # Integral = T * ∫ D(t'*T) * sin(...) dt' from 0 to 1
                # np.trapezoid computes ∫ f(x) dx, so we need to pass time_data
                if self.integration_method.lower() == "simpson":
                    psd[i] = (2.0 / T_period) * simpson(integrand, time_data, axis=1)
                else:
                    psd[i] = (2.0 / T_period) * np.trapezoid(
                        integrand, time_data, axis=1
                    )

        else:
            # Averaged or single cycle: D shape (n_spectra, n_wavenumbers)
            n_wavenumbers = D.shape[1]

            # Result array: (n_phi, n_wavenumbers)
            psd = np.zeros((n_phi, n_wavenumbers))

            # angle: (n_phi, n_spectra)
            angle = (
                harmonic * 2.0 * np.pi * time_data / T_period + phi_rad[:, np.newaxis]
            )  # (n_phi, n_spectra)
            # integrand: (n_phi, n_spectra, n_wavenumbers)
            integrand = D[np.newaxis, :, :] * np.sin(angle)[:, :, np.newaxis]

            # Integrate over spectra dimension (axis=1)
            if self.integration_method.lower() == "simpson":
                psd = (2.0 / T_period) * simpson(integrand, time_data, axis=1)
            else:
                psd = (2.0 / T_period) * np.trapezoid(integrand, time_data, axis=1)

        return psd

    def _fit(self, X, Y=None):
        """
        Fit the PSD model to X.

        Parameters
        ----------
        X : NDDataset or array-like
            Input data for PRS.
        Y : None
            Ignored, present for API compatibility.

        Returns
        -------
        self
            The fitted PSD instance.
        """
        # Normalize input to 3D
        data_3d, time_coord, cycle_coord, spectral_coord = self._normalize_input(X)

        # Store spectral coord for later use
        self._spectral_coord = spectral_coord

        # Optionally subtract mean along time axis
        if self.subtract_mean:
            # Subtract mean along the time (second) axis
            mean = np.mean(data_3d, axis=1, keepdims=True)
            data_3d = data_3d - mean

        # Compute transform matrix T (for matrix method)
        if self.method == "matrix":
            self._T = self._compute_T(time_coord)

        # Always average cycles before PSD
        # A shape: (n_spectra, n_wavenumbers)
        if self._n_cycles > 1:  # noqa: SIM108
            A = np.mean(data_3d, axis=0)  # Average over cycles
        else:
            A = data_3d[0]  # Single cycle

        # Compute PSD
        if self.method == "matrix":
            psd_data = self._compute_psd_matrix(A)
        else:
            psd_data = self._compute_psd_integration(A)

        # Create NDDataset for psd
        # Shape: (n_phi, n_wavenumbers)
        psd = NDDataset(psd_data, dims=["y", "x"])
        # Create fresh coords to avoid dimension mismatch
        y_coord = Coord(self._get_phi(), title="y", units="degrees")
        # Preserve spectral coordinate from input
        if spectral_coord is not None:
            x_coord = spectral_coord.copy()
        else:
            x_coord = Coord(
                np.arange(self._n_wavenumbers),
                title="wavenumber",
                units=None,
            )
        psd.set_coordset(y=y_coord, x=x_coord)

        psd.units = X.units if hasattr(X, "units") else None
        psd.title = f"PSD of {X.title if hasattr(X, 'title') else 'data'}"
        psd.history = "Created by SpectroChemPy PSD"

        self._prs = psd

        # Extract in_phase (phi=0°), quadrature (phi=90°), amplitude, phase
        self._extract_components()

        return self

    def _extract_components(self):
        """Extract in_phase, quadrature, amplitude, and phase from psd."""
        phi = self._get_phi()

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
        in_phase_data = self._prs.data[idx_0, :]
        quadrature_data = self._prs.data[idx_90, :]

        # Create NDDatasets
        in_phase = NDDataset(in_phase_data)
        in_phase.dims = ["x"]
        in_phase.set_coordset(x=self._prs.coordset["x"].copy())
        in_phase.units = self._prs.units
        in_phase.title = "In-phase component (0°)"
        in_phase.history = "Created by SpectroChemPy PSD"

        quadrature = NDDataset(quadrature_data)
        quadrature.dims = ["x"]
        quadrature.set_coordset(x=self._prs.coordset["x"].copy())
        quadrature.units = self._prs.units
        quadrature.title = "Quadrature component (90°)"
        quadrature.history = "Created by SpectroChemPy PSD"

        self._in_phase = in_phase
        self._quadrature = quadrature

        # Compute amplitude
        amplitude_data = np.sqrt(in_phase.data**2 + quadrature.data**2)
        amplitude = NDDataset(amplitude_data)
        amplitude.dims = in_phase.dims
        amplitude.set_coordset(**in_phase.coordset)
        amplitude.units = in_phase.units
        amplitude.title = "PSD Amplitude"
        amplitude.history = "Created by SpectroChemPy PSD"
        self._amplitude = amplitude

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
        self._phase = phase

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None):
        """
        Fit the PSD model on X.

        Parameters
        ----------
        X : NDDataset or array-like
            Input data for PRS.
        Y : None
            Ignored, present for API compatibility.

        Returns
        -------
        self : PSD
            The fitted PSD instance.
        """
        self._fitted = False

        # Set X (this triggers validation and preprocessing)
        self._X = X

        # Call internal fit
        self._fit(X, Y)

        self._fitted = True
        return self

    def transform(self, X):
        """
        Transform data X using the fitted PSD model.

        Parameters
        ----------
        X : NDDataset or array-like
            Input data for PRS.

        Returns
        -------
        np.ndarray
            Phase-resolved spectra (PRS) result.
        """
        if not self._fitted:
            raise NotFittedError("The fit method must be used before using transform()")

        # For PRS, transform re-computes with same parameters
        # but on new data
        old_X = self._X
        self._X = X
        self._fit(X, None)
        # Return the psd from this new fit
        result = self._prs
        # Restore original data
        self._X = old_X
        return result

    def fit_transform(self, X, **kwargs):
        """
        Fit the PSD model on X and return the PSD.

        Parameters
        ----------
        X : NDDataset or array-like
            Input data for PSD.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        NDDataset
            Phase-resolved spectra (PRS).
        """
        self.fit(X)
        return self.prs

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

    # ----------------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------------
    @property
    def T(self):
        """Return the transform matrix T."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get T")
        return self._T

    @property
    def prs(self):
        """Return the phase-resolved spectra (PRS)."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get prs")
        return self._prs

    @property
    def in_phase(self):
        """Return the in-phase component (phi=0°)."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get in_phase")
        return self._in_phase

    @property
    def quadrature(self):
        """Return the quadrature component (phi=90°)."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get quadrature")
        return self._quadrature

    @property
    def amplitude(self):
        """Return the amplitude (sqrt(in_phase² + quadrature²))."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get amplitude")
        return self._amplitude

    @property
    def phase(self):
        """Return the phase (atan2(quadrature, in_phase))."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get phase")
        return self._phase
