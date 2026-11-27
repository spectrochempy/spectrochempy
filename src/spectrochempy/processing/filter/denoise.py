import numpy as np
from scipy.signal import savgol_filter

from spectrochempy.application.application import error_
from spectrochempy.application.application import get_loglevel
from spectrochempy.application.application import info_
from spectrochempy.application.application import warning_

__all__ = ["denoise", "despike"]
__dataset_methods__ = __all__


def denoise(dataset, ratio=99.8, **kwargs):
    r"""
    Denoise the data using a PCA method.

    Work only on 2D dataset.

    Parameters
    ----------
    dataset : `NDDataset` or a ndarray-like object
        Input object. Must have two dimensions.
    ratio : `float`, optional, default: 99.8%
        Ratio of variance explained in \%. The number of components selected for
        reconstruction is chosen automatically  such that the amount of variance that
        needs to be explained is greater than the percentage specified by `ratio` .
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    `NDDataset`
        Denoised 2D dataset

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an
        integer it is equivalent to the usual `axis` numpy parameter.
    log_level : str, int or None, optional, default: None
        An explicit level changes global logging; None preserves it.

    """
    from spectrochempy.analysis.decomposition.pca import PCA

    if dataset.ndim != 2:
        error_("Only 2D dataset are supported")
        return dataset

    if ratio > 100.0 or ratio < 0.0:
        error_("ratio must be between 0 and 100")
        return dataset

    ratio = ratio / 100.0

    log_level = kwargs.pop("log_level", get_loglevel())
    dim = kwargs.pop("dim", -1)
    axis, _ = dataset.get_axis(dim, negative_axis=True)
    swapped = False
    if axis != -1:
        dataset = dataset._swapdims_without_history(axis, -1)
        swapped = True

    pca = PCA(n_components=ratio, svd_solver="full", log_level=log_level)
    pca.fit(dataset)
    info_(
        f"Number of components selected for reconstruction: {pca.n_components} "
        f"[n_observations={dataset.shape[0]}, ratio={ratio * 100: .2f}%]",
    )
    if pca.n_components < 3:
        warning_(
            f"The number of components ({pca.n_components}) selected for "
            f"reconstruction seems very low.\nYour likely to have a poor "
            f"reconstruction.\nTry to increase the ratio.",
        )
    data = pca.inverse_transform()
    if swapped:
        data = data._swapdims_without_history(-1, axis)

    return data


def _despike_neighbors(dataset, delta, dim):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite.")
    if np.iscomplexobj(dataset.data):
        raise ValueError("Neighbor despiking requires real data.")

    new = dataset.copy()
    axis, name = new.get_axis(dim)
    data = np.moveaxis(new.data, axis, 0)
    mask = np.moveaxis(np.broadcast_to(new.mask, new.shape), axis, 0)
    if len(data) >= 3:
        # Floating-point arithmetic also allows fractional replacements in integer data.
        data = data.astype(np.result_type(data.dtype, np.float64), copy=True)
        samples = np.ma.array(data, mask=mask | ~np.isfinite(data))
        previous, current, following = samples[:-2], samples[1:-1], samples[2:]
        local_mean = (previous + following) / 2
        residual = current - local_mean
        center = np.ma.median(residual, axis=0)
        noise = 1.4826 * np.ma.median(abs(residual - center), axis=0)
        excess = current - np.ma.maximum(previous, following)
        spikes = (excess > delta * noise).filled(False)
        data[1:-1][spikes] = local_mean.data[spikes]
        new._dtype = data.dtype
        new._data = np.moveaxis(data, 0, axis)

    new.history = f"Removed spikes using neighbors (dim={name}, delta={delta})"
    return new


def _despike_trace(X, size, delta, method):
    if X.size < 2 or np.all(X[0] == X):
        return
    if method == "katsumoto" and X.size < size:
        return
    eps = np.finfo(float).eps
    s = int((size - 1) / 2)
    if method == "katsumoto":
        A = savgol_filter(X, window_length=size, polyorder=2)

        diff = X - A
        std = delta * np.std(diff)

        select = abs(diff) >= std
        select = np.logical_or(select, np.roll(select, 1))
        select = np.logical_or(select, np.roll(select, -1))

        w = np.ones_like(X)
        w[select] = 0

        res = np.zeros_like(X)
        W = np.zeros_like(X) + eps
        r = range(-s, s + 1)
        for j in r:
            vj = np.roll(X, j)
            wj = np.roll(w, j)
            res += vj * wj
            W += wj

        A = res / W

        X[select] -= (X - A)[select]

    elif method == "whitaker":
        DX = np.zeros_like(X)
        DX[1:] = X[1:] - X[:-1]

        m = np.median(DX)
        M = np.median(np.abs(DX - m))
        Z = (DX - m) / M if M > 0 else np.where(m == DX, 0.0, np.inf)
        Z[0] = Z[-1] = delta + 1

        select = abs(Z) >= delta

        A = np.zeros_like(X)
        for i in [j for j, x in enumerate(select) if x]:
            indexes = np.arange(max(0, i - s), min(len(X), i + s))
            indexes = [index for index in indexes if not select[index]]
            if indexes != []:
                A[i] = np.mean(X[indexes])
            else:
                A[i] = X[i]

        X[select] -= (X - A)[select]


def _despike_output(source, corrected, return_mask):
    if not return_mask:
        return corrected
    valid = ~np.broadcast_to(source.mask, source.shape) & np.isfinite(source.data)
    changed = valid & (source.data != corrected.data)
    diagnostic = source.__class__(
        changed,
        dtype=np.bool_,
        dims=source.dims,
        coordset=source.coordset.copy() if source.coordset is not None else None,
        mask=source.mask,
        title="Points changed by despike",
        name=f"{source.name}_despike_changes",
    )
    diagnostic.history = (
        f"Correction diagnostic: {corrected.history_entries[-1]['message']}"
    )
    return corrected, diagnostic


def despike(dataset, size=9, delta=2, method="katsumoto", *, dim=-1, return_mask=False):
    """
    Remove spikes from the data.

    The `despike` methods can be used to remove cosmic ray peaks from a spectrum.

    The 'katsumoto' implementation (default) is based on the method is described
    in :cite:t:`katsumoto:2003`:

    * In the first step, the moving-average method is employed to detect the spike
      noise. The moving-average window should contain several data points along the
      abscissa that are larger than those of the spikes in the spectrum.
      If a window contains a spike, the value on the ordinate for the spike will show
      an anomalous deviation from the average for this window.
    * In the second step, each data point value identified as noise is replaced by the
      moving-averaged value.
    * In the third step, the moving-average process is applied to the new data set made
      by the second step.
    * In the fourth step, the spikes are identified by comparing the differences between
      the original spectra and the moving-averaged spectra calculated in the third step.

    As a result, the proposed method realizes the reduction of convex spikes.

    The 'whitaker' implementation is based on the method is described in :cite:t:`whitaker:2018`:

    * The spikes are detected when the zscore of the difference between consecutive intensities is larger than the delta
      parameter.
    * The spike intensities are replaced by the average of the intensities in a window around the spike, excluding the
      points that are spikes.


    The 'neighbors' method compares each sample with its two immediate
    neighbors along ``dim``, independently at every position on the other
    dimensions. A positive excursion above both neighbors is replaced by
    their mean if it exceeds ``delta`` times the scaled median absolute
    deviation of the residuals from the neighbor mean. Endpoints, triplets
    with masked or non-finite values are skipped. With zero estimated noise,
    any strictly positive excess above both neighbors qualifies. For a time series of spectra, choose
    the acquisition axis. Short traces may contain too few unaffected samples
    to estimate the noise reliably.

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset, left unchanged.
    size : int, optional, default: 9
        Size of the moving average window ('katsumoto' method) or the size of the window around the spike to estimate
        the intensities ('whitaker' method).
    delta : float, optional, default: 2
        Set the threshold for the detection of spikes.
    method : str, optional, default: 'katsumoto'
        The method to use. Can be 'katsumoto', 'whitaker', or 'neighbors'.
    dim : str or int, optional, default: -1
        Dimension used by all methods. Other dimensions are processed independently.
        'neighbors' uses exactly two immediate neighbors, independently of ``size``.
    return_mask : bool, optional, default: False
        Return ``(corrected, changed)`` instead of only the corrected dataset.
        ``changed`` is a dimensionless boolean NDDataset on the original coordinates,
        True exactly where a finite, unmasked value was modified. It preserves the
        input mask; masked and non-finite values have False underlying flags.
        A changed point is not proof of a cosmic ray: genuine bands can be modified.

    Returns
    -------
    corrected : NDDataset
        Corrected data, with original coordinates, units, mask and metadata.
    changed : NDDataset, optional
        Boolean correction diagnostic, only when ``return_mask=True``.

    Notes
    -----
    Masked and non-finite values remain unchanged. Katsumoto and Whitaker operate
    separately on contiguous valid runs; Katsumoto skips runs shorter than ``size``.
    Constant runs remain unchanged. Integer inputs allow fractional replacements
    only when every source value is exactly representable as float64; otherwise
    a ValueError is raised before processing, including for masked values.
    The existing Katsumoto circular window and Whitaker endpoint candidacy are
    retained within each run; inspect corrections near boundaries and gaps.
    For Katsumoto, size=3 leaves only rounding residuals after the quadratic fit
    and can yield unstable selection; use a larger window and inspect the result.
    Narrow spectral bands can be mistaken for spikes, especially with low thresholds
    or little noise. Compare corrected spectra and band areas with the original.

    """
    if method not in ("katsumoto", "whitaker", "neighbors"):
        raise ValueError(
            f"Unknown despike method {method!r}. "
            "Expected one of: katsumoto, whitaker, neighbors."
        )

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite.")
    if np.iscomplexobj(dataset.data):
        raise ValueError("Despiking requires real data.")
    if np.issubdtype(dataset.data.dtype, np.integer) and any(
        int(value) != int(float(value)) for value in dataset.data.flat
    ):
        raise ValueError("Integer data must be exactly representable as float64.")
    if method == "neighbors":
        return _despike_output(
            dataset, _despike_neighbors(dataset, delta, dim), return_mask
        )
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)):
        raise ValueError("size must be a positive integer.")
    if size < (3 if method == "katsumoto" else 1):
        raise ValueError(
            "size must be at least 3 for katsumoto and positive for whitaker."
        )

    new = dataset.copy()
    axis, name = dataset.get_axis(dim)
    dtype = new.data.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(float)
    data = np.moveaxis(dataset.data, axis, -1).astype(dtype, copy=True)
    mask = np.moveaxis(np.broadcast_to(dataset.mask, dataset.shape), axis, -1)
    for index in np.ndindex(data.shape[:-1]):
        trace = data[index]
        valid = ~mask[index] & np.isfinite(trace)
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for begin, end in edges.reshape(-1, 2):
            _despike_trace(trace[begin:end], size, delta, method)
    new._dtype = data.dtype
    new._data = np.moveaxis(data, -1, axis)
    new.history = (
        f"Removed spikes using {method} (size={size}, delta={delta}, dim={name})"
    )
    return _despike_output(dataset, new, return_mask)
