# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np
import scipy.signal
import traitlets as tr

import spectrochempy.utils.traits as mtr
from spectrochempy.extern.whittaker_smooth import whittaker_smooth as ws
from spectrochempy.processing._base._processingbase import ProcessingConfigurable
from spectrochempy.utils.decorators import signature_has_configurable_traits

__dataset_methods__ = [
    "savgol_filter",
    "savgol",
    "smooth",
    "whittaker",
]
__configurables__ = ["Filter"]
__all__ = __dataset_methods__ + __configurables__


def _detect_uniform_spacing(dataset, dim):
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

    Returns
    -------
    delta_signed : float or None
        The mean signed spacing when the coordinate is uniformly spaced,
        or ``None`` when detection fails.
    message : str or None
        Explanation when *delta_signed* is ``None``, else ``None``.
    """
    try:
        coord = dataset.coord(dim)
    except AttributeError:
        return None, "no coordinate available"

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


_common_see_also = """
See Also
--------
Filter : Define and apply filters/smoothers using various algorithms.
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
    :term:`n_features`).

    Various filters/smoothers can be applied to the data. The currently available
    filters are:

    - Moving average (`avg`)
    - Convolution filters (`han`, `hamming`, `bartlett`, `blackman`)
    - Savitzky-Golay filter (`savgol`)
    - Whittaker-Eilers filter (`whittaker`)

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    method : any of [``"avg"``, ``"han"``, ``"hamming"``, ``"bartlett"``, ``"blackman"``, ``"median"``, ``"savgol"``, ``"whittaker"``], optional, default: ``"savgol"``
        The filter method to be applied. By default, the Savitzky-Golay (savgol) filter is applied.
    size : `int`, optional, default: 5
        The size of the filter window. size must be a positive odd integer.
    order : `int`, optional, default: 2
        The order of the polynomial used to fit the data.
    deriv : `int`, optional, default: 0
        The order of the derivative to compute.
    lamb : `float`, optional, default: 10.0
        The smoothing parameter for the Whittaker-Eilers filter.
    cval : `float`, optional, default: 0.0
        The value to fill past the edges of the input if `mode` is ``'constant'``.

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
        "difference order of the penalized least squares.",
    ).tag(config=True, min=0)

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
        "if the coordinate is uniformly spaced.  On a non-uniform or "
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
    def _transform(self, X):
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
                    if delta_signed is not None:
                        delta_used = delta_signed
                        delta_source = "coordinate"
                    else:
                        delta_used = 1.0
                        delta_source = "fallback"
                        import warnings as _warnings

                        _warnings.warn(
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
                self._preserve_identity = False

        # Whittaker-Eilers filter
        # -----------------------
        elif self.method == "whittaker":
            data = np.apply_along_axis(ws, -1, X, self.lamb, self.order)

        return data


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
          automatically derived from the coordinate of the processed axis
          if the coordinate is uniformly spaced.  On a non-uniform or
          missing coordinate a warning is emitted and the index-based
          ``delta=1.0`` is used as a fallback.
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

    Notes
    -----
    When ``delta`` is ``None`` (the default), the sample spacing is
    detected from the coordinate and passed directly to
    ``scipy.signal.savgol_filter``.  The Savitzky-Golay algorithm is
    fundamentally index-based; the detected delta scales the derivative
    coefficients during the convolution.

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
