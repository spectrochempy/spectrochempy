# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import warnings
from numbers import Integral

import numpy as np
import scipy.ndimage
import scipy.signal
import traitlets as tr

import spectrochempy.utils.traits as mtr
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.extern.whittaker_smooth import whittaker_smooth as ws
from spectrochempy.processing._base._processingbase import ProcessingConfigurable
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.optional import import_optional_dependency

__dataset_methods__ = [
    "differentiate",
    "savgol_filter",
    "savgol",
    "smooth",
    "whittaker",
]
__configurables__ = ["Filter"]
__all__ = __dataset_methods__ + __configurables__


def _detect_uniform_spacing(dataset, dim, *, coordset=None):
    """
    Detect uniform spacing from the coordinate along *dim*.

    Uses the raw float64 storage (``coord._data``) to avoid precision loss
    from the ``Coord`` rounding layer.  All successive differences are
    compared to their mean with ``numpy.allclose(rtol=1e-10, atol=0)``.
    The mean signed delta is returned (not ``diffs[0]``).

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset whose coordinate is inspected.
    dim : int
        The resolved integer axis index.
    coordset : `CoordSet`, optional
        Original coordinates to inspect before masked samples were removed.

    Returns
    -------
    delta_signed : float or None
        The mean signed spacing when the coordinate is uniformly spaced,
        or ``None`` when detection fails.
    message : str or None
        Explanation when *delta_signed* is ``None``, else ``None``.
    """
    try:
        coord = dataset.coord(dim) if coordset is None else coordset[dataset.dims[dim]]
    except AttributeError:
        return None, "no coordinate available"

    return _uniform_spacing(coord)


def _uniform_spacing(coord):
    if coord is None:
        return None, "coordinate is None"

    if coord.is_masked or (
        hasattr(coord, "_mask")
        and isinstance(coord._mask, np.ndarray)
        and np.any(coord._mask)
    ):
        return None, "coordinate contains masked values"

    # Use coord._data (raw float64 storage) rather than coord.data.
    # The public .data property applies a rounding layer that truncates
    # float64 to ~4 significant digits, destroying the uniform-spacing
    # signal.  _data preserves the original precision.
    try:
        values = np.asarray(coord._data, dtype=float)
    except (TypeError, ValueError):
        return None, "coordinate is not numeric"

    if values.ndim != 1 or values.size < 2:
        return None, "coordinate has fewer than 2 points"

    if not np.all(np.isfinite(values)):
        return None, "coordinate contains non-finite values (NaN or Inf)"

    diffs = np.diff(values)

    if np.all(diffs == 0):
        return None, "coordinate is degenerate (all values identical)"

    mean_diff = np.mean(diffs)

    if not np.allclose(diffs, mean_diff, rtol=1e-10, atol=0):
        return None, "coordinate is not uniformly spaced"

    return float(mean_diff), None


_SPECTRAL_METHODS = {"butterworth", "notch", "gaussian", "wavelet"}


_common_see_also = """
See Also
--------
Filter : Define and apply filters/smoothers using various algorithms.
differentiate : Compute a smoothed derivative using a Savitzky-Golay filter.
smooth : Function to smooth data using various window filters.
savgol : Savitzky-Golay filter.
savgol_filter : Alias of `savgol`
whittaker : Whittaker-Eilers filter.
"""


# ======================================================================================
# Filter class processor
# ======================================================================================
@signature_has_configurable_traits
class Filter(ProcessingConfigurable):
    """
    Filters/smoothers processor.

    The filters can be applied to 1D datasets consisting in a single row
    with :term:`n_features` or to a 2D dataset with shape (:term:`n_observations`,
    :term:`n_features`). Butterworth, notch, Gaussian and wavelet filters also support
    higher-dimensional datasets. Physical settings use the active coordinate
    on axes with multiple coordinates.

    Various filters/smoothers can be applied to the data. The currently available
    filters are:

    - Moving average (`avg`)
    - Convolution filters (`han`, `hamming`, `bartlett`, `blackman`)
    - Savitzky-Golay filter (`savgol`)
    - Whittaker-Eilers filter (`whittaker`)
    - Zero-phase Butterworth low-pass filter (`butterworth`)
    - Zero-phase narrow band-stop filter (`notch`)
    - Gaussian convolution (`gaussian`)
    - Discrete wavelet denoising (`wavelet`)

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    method : any of [``"avg"``, ``"han"``, ``"hamming"``, ``"bartlett"``, ``"blackman"``, ``"median"``, ``"savgol"``, ``"whittaker"``, ``"butterworth"``, ``"notch"``, ``"gaussian"``, ``"wavelet"``], optional, default: ``"savgol"``
        The filter method to be applied. By default, the Savitzky-Golay (savgol) filter is applied.
    size : `int`, optional, default: 5
        The size of the filter window. size must be a positive odd integer.
    order : `int`, optional, default: 2
        For Savitzky-Golay, the order of the polynomial used to fit the data.
        For Butterworth, the positive filter order per pass.
    deriv : `int`, optional, default: 0
        The order of the derivative to compute.
    lamb : `float`, optional, default: 10.0
        The smoothing parameter for the Whittaker-Eilers filter.
    cval : `float`, optional, default: 0.0
        The value to fill past the edges of the input if `mode` is ``'constant'``.

    cutoff : `float`, optional
        Required for Butterworth. Frequency in cycles per unit of the selected
        coordinate. The combined forward-backward response is −3 dB at cutoff.
    period : `float`, optional
        Required for the notch filter. Period of the fringe in units of the
        selected coordinate.
    quality_factor : `float`, optional, default: 30.0
        Notch center frequency divided by the single-pass −3 dB bandwidth.
        Must exceed 0.5 and give a bandwidth below the Nyquist frequency.
    sigma : `float`, optional
        Required for Gaussian convolution. Standard deviation of the Gaussian.
    sigma_units : {"points", "coordinate"}, optional, default: "points"
        Interpret Gaussian sigma in sample points or in units of the selected
        coordinate. FWHM equals sqrt(8 log(2)) times sigma.

    wavelet : str, optional, default: "sym4"
        Discrete wavelet family and order, using PyWavelets names.
    wavelet_level : int or None, optional
        Positive decomposition level. The default is the smaller of three and
        the maximum level for the selected wavelet and axis length.
    wavelet_threshold : float or None, optional
        Nonnegative soft threshold in signal units. By default it is estimated
        separately for each spectrum from the median absolute deviation of
        the finest detail coefficients (MAD divided by 0.67449), multiplied
        by sqrt(2 log(n)), where n is the selected axis length.

    Notes
    -----
    These filters require finite, unmasked data. Butterworth, notch and
    coordinate-based Gaussian filtering require a uniformly spaced coordinate.
    Descending coordinates are supported. Butterworth and notch use odd
    extension with SciPy's default padding and require more than the padding
    length in samples along the selected axis. The padding is three times
    the Butterworth order plus three, or nine points for the notch.
    IIR settings whose response cannot be resolved in floating point are rejected.
    Gaussian convolution uses the configured `mode` and `cval`.
    Inputs outside SciPy's supported floating-point dtypes are converted
    to double precision. Overflow during conversion is rejected.
    The default `interp` selects SciPy's `reflect` mode for this method.

    Wavelet denoising uses symmetric extension and sample indices along the
    selected axis. It soft-thresholds detail coefficients and preserves the
    approximation coefficients. It accepts real data and computes in double
    precision. It requires the optional dependency installed with ``pip install "spectrochempy[wavelets]"``.

    See Also
    --------
    smooth : Function to smooth data using various window filters.
    savgol : Savitzky-Golay filter.
    savgol_filter : Alias of `savgol`
    whittaker : Whittaker-Eilers filter.
    """

    method = tr.Enum(
        [
            "avg",
            "han",
            "hamming",
            "bartlett",
            "blackman",
            "median",
            "savgol",
            "whittaker",
            "butterworth",
            "notch",
            "gaussian",
            "wavelet",
        ],
        default_value="savgol",
        help="The filter method to be applied. By default, "
        "the Savitzky-Golay (savgol) filter is applied.",
    ).tag(config=True)

    size = mtr.PositiveOddInteger(
        default_value=5,
        help="The size of the filter window.size must be a positive odd integer.",
    ).tag(config=True)

    order = tr.Integer(
        default_value=2,
        help="The order of the polynomial used to fit the data"
        "in the case of the Savitzky-Golay (savgol) filter. "
        "`order` must be less than size.\n"
        "In the case of the Whittaker-Eilers filter, order is the "
        "difference order of the penalized least squares. "
        "For Butterworth it is the positive filter order per pass.",
    ).tag(config=True, min=0)

    cutoff = tr.Float(
        None, allow_none=True, help="Butterworth cutoff in cycles per coordinate unit."
    ).tag(config=True)

    period = tr.Float(
        None, allow_none=True, help="Notch period in coordinate units."
    ).tag(config=True)

    quality_factor = tr.Float(
        30.0, help="Notch center frequency divided by single-pass bandwidth."
    ).tag(config=True)

    sigma = tr.Float(None, allow_none=True, help="Gaussian standard deviation.").tag(
        config=True
    )

    sigma_units = tr.Enum(
        ["points", "coordinate"],
        default_value="points",
        help="Interpret Gaussian sigma in points or in coordinate units.",
    ).tag(config=True)

    wavelet = tr.Unicode("sym4", help="Discrete wavelet name.").tag(config=True)

    wavelet_level = tr.Integer(
        None, allow_none=True, help="Discrete wavelet decomposition level."
    ).tag(config=True)

    wavelet_threshold = tr.Float(
        None, allow_none=True, help="Soft threshold in signal units."
    ).tag(config=True)

    deriv = tr.Integer(
        default_value=0,
        help="The order of the derivative to compute in the case of "
        "the Savitzky-Golay (savgol) filter. This must be a "
        "non-negative integer. The default is 0, which means to "
        "filter the data without differentiating.",
    ).tag(config=True, min=0)

    lamb = tr.Float(
        default_value=1.0,
        help=r"Smoothing/Regularization parameter. The larger `lamb`, the smoother "
        "the data.",
    ).tag(config=True)

    delta = tr.Float(
        default_value=None,
        allow_none=True,
        help="The signed sample spacing passed to ``scipy.signal.savgol_filter``. "
        "This is only used if deriv > 0.\n\n"
        "When ``None`` (the default), the signed spacing "
        "is automatically derived from the coordinate of the processed axis "
        "if the retained samples are uniformly spaced, otherwise from the "
        "original coordinate if it is uniform. On a non-uniform or "
        "missing coordinate a warning is emitted and the index-based "
        "``delta=1.0`` is used as a fallback.\n\n"
        "When set to a numeric value, that value is passed directly to "
        "SciPy with its sign.  No unit-based correction (``_reversed``) "
        "is applied.  For a descending coordinate, supply a negative "
        "``delta`` if the derivative should follow the physical axis.",
    ).tag(config=True)

    mode = tr.Enum(
        ["mirror", "constant", "nearest", "wrap", "interp"],
        default_value="interp",
        help="""
The type of extension to use for the padded signal to which the filter is applied.

* When mode is ‘constant’, the padding value is given by `cval`.
* When the ‘interp’ mode is selected (the default), no extension is used.
  Instead, a polynomial of degree `order` is fit to the last `size` values
  of the edges, and this polynomial is used to evaluate the last size // 2
  output values.
* When mode is ‘nearest’, the last size values are repeated.
* When mode is ‘mirror’, the padding is created by reflecting the signal about the end
  of the signal.
* When mode is ‘wrap’, the signal is wrapped around on itself to create the padding.

See `scipy.signal.savgol_filter` for more details on ‘mirror’, ‘constant’, ‘wrap’,
and ‘nearest’.
""",
    ).tag(config=True)

    cval = tr.Float(
        default_value=0.0,
        help="Value to fill past the edges of the input if `mode` is ‘constant’. ",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        log_level="WARNING",
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def transform(self, dataset, dim=-1):
        """
        Apply the configured filter along a dimension of the dataset.

        Parameters
        ----------
        dataset : `NDDataset`
            The dataset to filter.
        dim : int or str, optional, default: -1
            A dimension name or an axis index.

        Returns
        -------
        `NDDataset`
            The filtered dataset.

        """
        if self.method not in _SPECTRAL_METHODS:
            return super().transform(dataset, dim=dim)

        self._transformed = False
        if dataset.ndim == 0:
            raise ValueError(f"{self.method} filtering requires a dimension")
        if np.any(dataset.mask):
            raise ValueError(f"{self.method} filtering requires unmasked data")
        if not np.all(np.isfinite(dataset.data)):
            raise ValueError(f"{self.method} filtering requires finite data")

        axis, dimension = dataset.get_axis(dim, negative_axis=True)
        data, parameters = self._spectral_transform(dataset, axis, dimension)
        with np.errstate(over="ignore", under="ignore"):
            result = dataset.copy(keepname=True)
        result._dtype = data.dtype
        result._data = data
        result._append_history_entry(
            operation=self.method,
            parameters={"dimension": dimension, "axis": axis, **parameters},
            message=f"Applied {self.method} filter on dimension {dimension}",
        )
        self._transformed = True
        return result

    def _spectral_transform(self, dataset, axis, dimension):
        if self.method == "wavelet":
            return self._wavelet_transform(dataset.data, axis)

        parameter = {"butterworth": "cutoff", "notch": "period", "gaussian": "sigma"}[
            self.method
        ]
        value = getattr(self, parameter)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{parameter} must be finite and positive")

        data = dataset.data
        gaussian = self.method == "gaussian"
        physical_units = not gaussian or self.sigma_units == "coordinate"
        spacing = 1.0
        parameters = {parameter: value}
        if physical_units:
            coord = getattr(dataset, dimension, None)
            if isinstance(coord, CoordSet):
                coord = coord.default
            delta, message = _uniform_spacing(coord)
            if delta is None or not np.isfinite(delta) or delta == 0:
                raise ValueError(
                    f"{self.method} filtering requires a uniformly spaced coordinate"
                    f" ({message or 'invalid spacing'})"
                )
            spacing = abs(delta)
            parameters.update(
                spacing=spacing,
                coordinate_units=str(coord.units) if coord.units is not None else None,
            )

        if gaussian:
            sigma_points = value / spacing
            if not np.isfinite(sigma_points) or sigma_points <= 0:
                raise ValueError("Gaussian sigma in points must be finite and positive")
            mode = "reflect" if self.mode == "interp" else self.mode
            parameters.update(
                sigma_units=self.sigma_units, sigma_points=sigma_points, mode=mode
            )
            if mode == "constant":
                if not np.isfinite(self.cval):
                    raise ValueError("Gaussian cval must be finite in constant mode")
                parameters["cval"] = self.cval

            truncate = 4.0
            if sigma_points < 0.5 / truncate:
                return data.copy(), parameters
            if data.dtype.type not in (
                np.float32,
                np.float64,
                np.complex64,
                np.complex128,
            ):
                dtype = np.complex128 if np.iscomplexobj(data) else np.float64
                with np.errstate(over="ignore", under="ignore"):
                    data = data.astype(dtype)
                if not np.isfinite(data).all():
                    raise ValueError(
                        "Gaussian input exceeds the double-precision range"
                    )
            return (
                scipy.ndimage.gaussian_filter1d(
                    data,
                    sigma_points,
                    axis=axis,
                    mode=mode,
                    cval=self.cval,
                    truncate=truncate,
                ),
                parameters,
            )

        nyquist_fraction = (
            2.0 * spacing * value
            if self.method == "butterworth"
            else 2.0 * spacing / value
        )
        if not 0 < nyquist_fraction < 1:
            raise ValueError(f"{parameter} must give a frequency below Nyquist")

        if self.method == "butterworth":
            if self.order < 1:
                raise ValueError("Butterworth order must be positive")
            padlen = 3 * (self.order + 1)
            factor = (np.sqrt(2.0) - 1.0) ** (-0.5 / self.order)
            critical = (2.0 / np.pi) * np.arctan(
                factor * np.tan(np.pi * nyquist_fraction / 2.0)
            )
            parameters.update(order=self.order, critical_nyquist_fraction=critical)
        else:
            if not np.isfinite(self.quality_factor) or self.quality_factor <= 0.5:
                raise ValueError("quality_factor must be finite and greater than 0.5")
            if nyquist_fraction / self.quality_factor >= 1:
                raise ValueError("quality_factor must give a bandwidth below Nyquist")
            padlen = 9
            parameters["quality_factor"] = self.quality_factor

        if data.shape[axis] <= padlen:
            raise ValueError(
                f"{self.method} filtering requires at least {padlen + 1} "
                f"samples along dimension {dimension!r}"
            )
        if self.method == "butterworth":
            sos = scipy.signal.butter(self.order, critical, output="sos")
            target_gain = 2.0**-0.25
        else:
            numerator, denominator = scipy.signal.iirnotch(
                nyquist_fraction, self.quality_factor
            )
            sos = np.array([np.concatenate((numerator, denominator))])
            target_gain = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            _, response = scipy.signal.sosfreqz(
                sos, worN=[0.0, np.pi * nyquist_fraction]
            )
        if not np.allclose(np.abs(response), [1.0, target_gain], rtol=1e-4, atol=1e-4):
            raise ValueError(
                f"{self.method} settings are outside the numerically reliable range"
            )

        parameters.update(padtype="odd", padlen=padlen)
        return (
            scipy.signal.sosfiltfilt(sos, data, axis=axis, padlen=padlen),
            parameters,
        )

    def _wavelet_transform(self, data, axis):
        pywt = import_optional_dependency(
            "pywt", extra='Install it with pip install "spectrochempy[wavelets]".'
        )
        if np.iscomplexobj(data):
            raise ValueError("wavelet filtering requires real data")
        with np.errstate(over="ignore", under="ignore"):
            data = data.astype(np.float64, copy=False)
        if not np.isfinite(data).all():
            raise ValueError("wavelet input exceeds the double-precision range")
        wavelet = pywt.Wavelet(self.wavelet)
        size = data.shape[axis]
        maximum_level = pywt.dwt_max_level(size, wavelet.dec_len)
        if maximum_level == 0:
            raise ValueError(
                f"wavelet {wavelet.name!r} requires at least "
                f"{2 * (wavelet.dec_len - 1)} samples along the selected dimension"
            )
        level = self.wavelet_level
        if level is None:
            level = min(3, maximum_level)
        if not 1 <= level <= maximum_level:
            raise ValueError(f"wavelet_level must be between 1 and {maximum_level}")

        threshold = self.wavelet_threshold
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
            raise ValueError("wavelet_threshold must be finite and nonnegative")

        mode = "symmetric"
        coefficients = pywt.wavedec(data, wavelet, mode=mode, level=level, axis=axis)
        rule = "fixed"
        if threshold is None:
            finest = coefficients[-1]
            center = np.median(finest, axis=axis, keepdims=True)
            mad = np.median(np.abs(finest - center), axis=axis, keepdims=True)
            threshold = mad * (np.sqrt(2 * np.log(size)) / 0.6744897501960817)
            if not np.isfinite(threshold).all():
                raise ValueError(
                    "wavelet threshold exceeds the supported numeric range"
                )
            rule = "universal_mad"

        details = [
            np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            for detail in coefficients[1:]
        ]
        result = pywt.waverec(
            [coefficients[0], *details], wavelet, mode=mode, axis=axis
        )
        extent = [slice(None)] * data.ndim
        extent[axis] = slice(size)
        result = result[tuple(extent)]
        if not np.isfinite(result).all():
            raise ValueError(
                "wavelet reconstruction exceeds the supported numeric range"
            )
        effective_threshold = (
            threshold.squeeze(axis=axis).tolist()
            if rule == "universal_mad"
            else threshold
        )
        return result, {
            "wavelet": wavelet.name,
            "level": level,
            "mode": mode,
            "threshold_mode": "soft",
            "threshold_rule": rule,
            "threshold": effective_threshold,
        }

    def _transform(self, X):
        self._history_delta = None
        kwargs = {  # param for avg and convolution filters
            "axis": self._dim,
            "mode": "reflect" if self.mode == "interp" else self.mode,
            "cval": self.cval,
        }

        # Reset the output-title annotation: only the Savitzky-Golay derivative
        # path sets it, so every other method (and deriv=0) leaves the title as
        # that of the input data.
        self._output_title_suffix = None
        self._preserve_identity = True
        # Clear any stale dynamic-unit override from a previous call.
        self._output_units = None

        # smooth with moving average
        # --------------------------
        if self.method == "avg":
            data = scipy.ndimage.uniform_filter1d(X, self.size, **kwargs)

        # Convolution filters
        # -------------------
        elif self.method in ["han", "hamming", "bartlett", "blackman"]:
            win = scipy.signal.get_window(self.method, self.size, fftbins=False)
            win = win / np.sum(win)
            data = scipy.ndimage.convolve1d(X, win, **kwargs)

        # Median filter
        # -------------
        elif self.method == "median":
            if "axis" in kwargs:
                axis = kwargs.pop("axis")
            if axis in (-2, 0):
                size = (self.size, 1)
            elif axis in (-1, 1):
                size = (1, self.size)
            data = scipy.ndimage.median_filter(X, size=size, **kwargs)

        # Savitzky-Golay filter
        # ---------------------
        elif self.method == "savgol":
            delta_used = self.delta
            # Track delta provenance for unit propagation.
            #   "irrelevant"   -> deriv == 0 (smoothing)
            #   "coordinate"   -> auto-detected from uniform coordinate
            #   "explicit"     -> user-provided numeric delta
            #   "fallback"     -> irregular/missing coord, delta=1.0
            delta_source = "irrelevant"

            if self.delta is None:
                if self.deriv:
                    delta_signed, msg = _detect_uniform_spacing(
                        self._X,
                        self._dim,
                    )
                    if delta_signed is None and np.any(self._X_mask):
                        delta_signed, msg = _detect_uniform_spacing(
                            self._X,
                            self._dim,
                            coordset=self._X_coordset,
                        )
                    if delta_signed is not None:
                        delta_used = delta_signed
                        delta_source = "coordinate"
                    else:
                        delta_used = 1.0
                        delta_source = "fallback"
                        warnings.warn(
                            f"Savitzky-Golay derivative requested but {msg}. "
                            "Falling back to index-based delta=1.0. "
                            "To obtain physically scaled derivatives, provide "
                            "a uniformly spaced coordinate or set delta explicitly.",
                            stacklevel=2,
                        )
                else:
                    # deriv=0: delta is irrelevant; scipy needs a float
                    delta_used = 1.0
            else:
                delta_source = "explicit"

            kwargs = {
                "axis": self._dim,
                "deriv": self.deriv,
                "delta": delta_used,
                "mode": self.mode,
                "cval": self.cval,
            }
            data = scipy.signal.savgol_filter(X, self.size, self.order, **kwargs)
            self._history_delta = delta_used

            # Propagate units for derivative paths that claim physical scaling.
            # Smoothing (deriv=0) and index fallbacks keep source units.
            if self.deriv > 0 and delta_source in ("coordinate", "explicit"):
                coord = self._X.coord(self._dim)
                coord_units = coord.units if coord is not None else None
                source_units = self._X.units
                if source_units is not None and coord_units is not None:
                    self._output_units = source_units / coord_units**self.deriv
                elif source_units is not None:
                    # Coordinate has no unit: keep source units (U3)
                    self._output_units = source_units
                else:
                    # Source has no unit: result has no unit (U5)
                    self._output_units = None

            # Annotate the output title so a derivative quantity is clearly
            # identified.  Coordinates are preserved by the output wrapping.
            if self.deriv:
                ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(
                    self.deriv, f"{self.deriv}th"
                )
                self._output_title_suffix = f"({ordinal} derivative)"

        # Whittaker-Eilers filter
        # -----------------------
        elif self.method == "whittaker":
            data = np.apply_along_axis(ws, self._dim, X, self.lamb, self.order)

        return data

    def _processing_history_entry(self):
        dimension = self._X.dims[self._dim]
        parameters = {
            "dimension": dimension,
            "axis": self._dim,
        }
        labels = {
            "avg": "moving-average",
            "han": "Hann",
            "hamming": "Hamming",
            "bartlett": "Bartlett",
            "blackman": "Blackman",
            "median": "median",
            "savgol": "Savitzky-Golay",
            "whittaker": "Whittaker-Eilers",
        }
        if self.method == "whittaker":
            parameters.update(lamb=self.lamb, order=self.order)
        else:
            parameters.update(size=self.size)
            if self.method == "savgol":
                parameters.update(order=self.order, deriv=self.deriv, mode=self.mode)
                if self.deriv:
                    parameters["delta"] = self._history_delta
                if self.mode == "constant":
                    parameters["cval"] = self.cval
            elif self.method in {
                "avg",
                "han",
                "hamming",
                "bartlett",
                "blackman",
                "median",
            }:
                parameters["mode"] = "reflect" if self.mode == "interp" else self.mode
                if self.mode == "constant":
                    parameters["cval"] = self.cval
        return {
            "operation": self.method,
            "parameters": parameters,
            "message": f"Applied {labels[self.method]} filter on dimension {dimension}",
        }


# ======================================================================================
# API / NDDataset functions
# ======================================================================================
# Instead of using directly the Filter class, we provide here some functions
# which are eventually more user-friendly and which can be used directly on NDDataset or
# called from the API.

# --------------------------------------------------------------------------------------


def smooth(dataset, size=5, window="avg", dim=-1, **kwargs):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled kernel window with the signal.

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset to smooth.
    size : `int`, optional, default: 5
        The size of the smoothing window.
    window : `str`, optional, default:'flat'
        The type of window from 'flat' or 'avg', 'han' or 'hanning', 'hamming',
        'bartlett', 'blackman'.
        `avg` window will produce a moving average smoothing.
    dim : `int` or `str`, optional, default: -1
        Axis along which to apply the filter.  Accepts a dimension name
        (e.g. ``"x"``) or an integer index (e.g. ``-1`` for the last axis).
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the filter.

    Returns
    -------
    `NDDataset`
        Smoothed data.

    Other Parameters
    ----------------
    mode : `str`, optional, default: 'nearest'
        The mode parameter determines how the array borders are handled.
    cval : `float`, optional, default: 0.0
        Value to fill past edges of input if mode is 'constant'.
    log_level : `str`, optional, default: 'WARNING'
        The log level for the filter.

    See Also
    --------
    Filter : Filter processing.

    """
    if window in ["flat", "avg", "han", "hanning", "hamming", "bartlett", "blackman"]:
        if window == "flat":
            window = "avg"
        if window == "hanning":
            window = "han"

        return Filter(method=window, size=size, **kwargs).transform(dataset, dim=dim)
    raise ValueError(
        f"Window type '{window}' is not supported. "
        f"Supported types are 'flat' or 'avg', 'han' or 'hanning', 'hamming', "
        f"'bartlett', 'blackman'.",
    )


# --------------------------------------------------------------------------------------
def differentiate(
    dataset,
    derivative_order=1,
    *,
    size=5,
    polynomial_order=2,
    delta=None,
    dim=-1,
    mode="interp",
    cval=0.0,
):
    """
    Differentiate a spectrum using a Savitzky-Golay polynomial filter.

    This is a discoverable differentiation interface to `savgol`; it computes
    a smoothed numerical derivative and does not use a separate finite-difference
    implementation.

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset to differentiate.
    derivative_order : `int`, optional, default: 1
        Order of the derivative. This maps to the ``deriv`` parameter of
        `savgol`. It must be a positive integer not greater than
        ``polynomial_order``.
    size : `int`, optional, default: 5
        Size of the Savitzky-Golay window. It must be a positive odd integer.
    polynomial_order : `int`, optional, default: 2
        Order of the polynomial fitted within each window. This maps to the
        ``order`` parameter of `savgol` and must be less than ``size``.
    delta : `float` or ``None``, optional, default: ``None``
        Signed sample spacing. When ``None``, `savgol` derives it from a
        uniformly spaced coordinate, preferring the retained samples and then
        the original axis before masked samples were removed. If neither is
        uniform, `savgol` warns and falls back to index-based spacing.
    dim : `int` or `str`, optional, default: -1
        Axis along which to differentiate. A dimension name such as ``"x"``
        or an integer axis index can be used.
    mode : `str`, optional, default: ``"interp"``
        Extension mode passed to the Savitzky-Golay filter.
    cval : `float`, optional, default: 0.0
        Padding value used when ``mode="constant"``.

    Returns
    -------
    `NDDataset`
        Smoothed derivative with the metadata produced by the equivalent
        `savgol` call.

    See Also
    --------
    savgol : General Savitzky-Golay filtering and differentiation interface.

    Notes
    -----
    The call ``differentiate(dataset, derivative_order=n, size=w,
    polynomial_order=p)`` is equivalent to ``savgol(dataset, deriv=n,
    size=w, order=p)``. Coordinate spacing, sign, units, masks, title,
    history, dimension selection, and numerical validation follow `savgol`
    unchanged. This facade additionally requires a positive derivative order
    no greater than the polynomial order.

    With masked data, windows spanning a masked gap can produce unreliable
    derivatives at nearby unmasked points. Combining regular subsampling
    with masked blocks can also give incorrect automatic scaling away from
    the gaps; supply the signed spacing of the retained samples as ``delta``
    in that case. See the masked-data notes in `savgol` for details.

    Examples
    --------
    >>> first_derivative = scp.differentiate(  # doctest: +SKIP
    ...     dataset,
    ...     derivative_order=1,
    ...     size=15,
    ...     polynomial_order=3,
    ... )
    """
    if (
        isinstance(derivative_order, bool)
        or not isinstance(derivative_order, Integral)
        or derivative_order < 1
    ):
        raise ValueError("derivative_order must be a positive integer")
    if (
        isinstance(polynomial_order, Integral)
        and not isinstance(polynomial_order, bool)
        and derivative_order > polynomial_order
    ):
        raise ValueError("derivative_order must not exceed polynomial_order")

    return savgol(
        dataset,
        size=size,
        order=polynomial_order,
        dim=dim,
        delta=delta,
        deriv=derivative_order,
        mode=mode,
        cval=cval,
    )


# --------------------------------------------------------------------------------------
def savgol(dataset, size=5, order=2, dim=-1, delta=None, **kwargs):
    """
    Savitzky-Golay filter.

    Wrapper of scpy.signal.savgol(). See the documentation of this function for more
    details.

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset to filter.
    size : `int`, optional, default: 5
        The size of the smoothing window.
    order : `int`, optional, default: 2
        The order of the polynomial used to fit the data. `order` must be less
        than size.
    dim : `int` or `str`, optional, default: -1
        Axis along which to apply the filter.  Accepts a dimension name
        (e.g. ``"x"``) or an integer index (e.g. ``-1`` for the last axis).
    delta : `float` or ``None``, optional, default: ``None``
        Sample spacing passed to ``scipy.signal.savgol_filter``.

        * ``None`` (default) — when ``deriv > 0``, the signed spacing is
          automatically derived from the retained samples of the processed
          axis if they are uniformly spaced. Otherwise, the original
          coordinate before masked samples were removed is used if it is
          uniform. If neither coordinate is uniform or no coordinate is
          available, a warning is emitted and the index-based ``delta=1.0``
          is used as a fallback.
        * A numeric value — passed directly to SciPy with its sign.  The
          value is interpreted in the current unit of the selected
          coordinate.  No unit-based correction (``_reversed``) is applied.
          For a descending coordinate, supply a negative ``delta`` if the
          derivative should follow the physical axis.

        .. versionchanged:: 0.12.5
           Default changed from ``1.0`` to ``None`` (auto-detect).

        .. versionchanged:: 0.12.5
           Explicit ``delta`` is now passed to SciPy with its sign.
           The former ``_reversed`` unit-based correction is no longer
           applied when ``delta`` is explicitly provided.

    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the filter.

    Returns
    -------
    `NDDataset`
        Smoothed data.

    Other Parameters
    ----------------
    deriv : `int`, optional, default: 0
        The order of the derivative to compute.
    mode : `str`, optional, default: 'nearest'
        The mode parameter determines how the array borders are handled.
    cval : `float`, optional, default: 0.0
        Value to fill past edges of input if mode is 'constant'.
    log_level : `str`, optional, default: 'WARNING'
        The log level for the filter.

    See Also
    --------
    Filter : Filter processing.
    differentiate : Discoverable interface for Savitzky-Golay derivatives.

    Notes
    -----
    Smoothing and derivative outputs preserve the source dataset name and
    retain its history, appending one structured ``savgol`` entry per call.
    This also applies to `savgol_filter`, `differentiate`, and
    `Filter.transform`.
    Derivative titles identify the derivative order, and units follow the
    scaling rules below.

    When ``delta`` is ``None`` (the default), the sample spacing is
    detected from the coordinate and passed directly to
    ``scipy.signal.savgol_filter``.  The Savitzky-Golay algorithm is
    fundamentally index-based; the detected delta scales the derivative
    coefficients during the convolution.

    **Masked data.** Masked rows and columns are removed before filtering,
    and the input mask is restored afterwards. Automatic spacing detection
    first uses the retained samples if they are uniformly spaced (including
    regular subsampling). Otherwise it uses the original coordinate if that
    is uniform, so a masked block does not force index-based scaling.
    If neither coordinate is uniform, the index-based fallback is used.

    Using the original spacing assumes that masking creates gaps without
    changing the sampling step elsewhere. Masks combining regular subsampling
    with a masked block can therefore give incorrectly scaled derivatives
    even far from the gaps, without a warning. In that case, set ``delta``
    explicitly to the signed spacing of the retained samples outside the
    gaps (for example, twice the original step when every other sample is
    removed). This corrects the scaling but not the effects of crossing gaps.

    Removing a masked block joins its two sides in the filtered series.
    Windows spanning this gap can therefore give unreliable values at
    unmasked points near the gap, even with physical units and spacing.
    Interior convolution windows can affect up to ``size // 2`` retained
    points on each side of a gap.
    The output mask is not expanded to cover these points. This also applies
    to smoothing and to an explicit ``delta``. For ``mode="interp"``,
    edge polynomial fits use a full window and can also span a nearby gap.

    When ``delta`` is a numeric value, it is passed to SciPy exactly as
    provided, with its sign.  The coordinate units (``cm⁻¹``, ``ppm``,
    etc.) have no effect on the result in this case.  The user is
    responsible for the sign convention.

    **Units.**  For ``deriv > 0`` with a physically scaled delta
    (auto-detected from a uniform coordinate or explicitly provided
    when the coordinate carries units), the output units are
    ``source_units / coordinate_units**deriv``.  For example, a first
    derivative of absorbance with respect to ``cm⁻¹`` yields
    ``absorbance·cm``.  Smoothing (``deriv=0``) and index-based
    fallbacks preserve the source units unchanged.  If the source has
    no units, the result has no units regardless of the coordinate.

    .. versionchanged:: 0.12.5
       Units are now propagated for physically scaled derivative
       paths.  Smoothing and index fallbacks keep source units.

    """
    return Filter(
        method="savgol", size=size, order=order, delta=delta, **kwargs
    ).transform(
        dataset,
        dim=dim,
    )


def savgol_filter(*args, **kwargs):
    """
    Savitzky-Golay filter.

    Alias of `savgol`.
    """
    return savgol(*args, **kwargs)


def whittaker(dataset, lamb=1.0, order=2, dim=-1, **kwargs):
    """
    Smooth the data using the Whittaker smoothing algorithm.

    This implementation based on the work by :cite:t:`eilers:2003` uses sparse matrices
    enabling high-speed processing of large input vectors.

    Copyright M. H. V. Werts, 2017 (see LICENSES/WITTAKER_SMOOTH_LICENSE.rst)

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset to smooth.
    lamb : `float`, optional, default: 1.0
        The smoothing parameter. Larger values make the result smoother.
    order : `int`, optional, default: 2
        The difference order of the penalized least-squares.
    dim : `int` or `str`, optional, default: -1
        Axis along which to apply the filter.  Accepts a dimension name
        (e.g. ``"x"``) or an integer index (e.g. ``-1`` for the last axis).
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to the filter.

    Returns
    -------
    `NDdataset`
        Smoothed data.

    Other Parameters
    ----------------
    log_level : `str`, optional, default: 'WARNING'
        The log level for the filter.

    See Also
    --------
    Filter : Filter processing.

    """
    return Filter(method="whittaker", lamb=lamb, order=order, **kwargs).transform(
        dataset,
        dim=dim,
    )
