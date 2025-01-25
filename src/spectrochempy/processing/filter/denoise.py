import numpy as np
from scipy.signal import savgol_filter

from spectrochempy.application import error_, info_, warning_
from spectrochempy.core import get_loglevel

__dataset_methods__ = ["denoise", "despike"]
__all__ = __dataset_methods__


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
    log_level : int, optional, default: "WARNING"
        Set the logging level for the method.
    """
    from spectrochempy.analysis.decomposition.pca import PCA

    if dataset.ndim != 2 and dataset.shape[0] > 1:
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
        dataset = dataset.swapdims(axis, -1)
        swapped = True

    pca = PCA(n_components=ratio, svd_solver="full", log_level=log_level)
    pca.fit(dataset)
    info_(
        f"Number of components selected for reconstruction: {pca.n_components} "
        f"[n_observations={dataset.shape[0]}, ratio={ratio * 10: .2f}%]"
    )
    if pca.n_components < 3:
        warning_(
            f"The number of components ({pca.n_components}) selected for "
            f"reconstruction seems very low.\nYour likely to have a poor "
            f"reconstruction.\nTry to increase the ratio."
        )
    data = pca.inverse_transform()
    if swapped:
        data = data.swapdims(-1, axis)

    return data


def despike(dataset, size=9, delta=2, method="katsumoto"):
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


    Parameters
    ----------
    dataset : `NDDataset` or a ndarray-like object
        Input object.
    size : int, optional, default: 9
        Size of the moving average window ('katsumoto' method) or the size of the window around the spike to estimate
        the intensities ('whitaker' method).
    delta : float, optional, default: 2
        Set the threshold for the detection of spikes.
    method : str, optional, default: 'katsumoto'
        The method to use. Can be 'katsumoto' or 'whitaker'

    Returns
    -------
    `NDdataset`
        The despike dataset
    """

    new = dataset.copy()

    # machine epsilon
    eps = np.finfo(float).eps

    s = int((size - 1) / 2)

    for k, X in enumerate(new.data):
        X = X.squeeze()

        if method == "katsumoto":
            # 1) first step : savgol filter
            A = savgol_filter(X, window_length=size, polyorder=2)

            # 2 and 3) second and third step : detect spike and replace spkike by the moving
            # average and the new data are smoothed again
            diff = X - A
            std = delta * np.std(diff)

            # spike should have a large variation with respect to the std of the difference
            select = abs(diff) >= std
            select = np.logical_or(select, np.roll(select, 1))
            select = np.logical_or(select, np.roll(select, -1))

            # compute weights
            w = np.ones_like(X)
            w[select] = 0

            # now we must calculate the weighted average, but for efficiency we will compute it
            # only around where spike peaks are, the other part should be unchanged.

            res = np.zeros_like(X)
            W = np.zeros_like(X) + eps  # to avoid division by zero
            r = range(-s, s + 1)
            for j in r:
                vj = np.roll(X, j)
                wj = np.roll(w, j)
                res += vj * wj
                W += wj

            A = res / W

            # 4) compare with original to remove spike peaks
            X[select] -= (X - A)[select]

        elif method == "whitaker":
            # 1) detrended difference series
            DX = np.zeros_like(X)
            DX[1:] = X[1:] - X[:-1]

            # zscore
            m = np.median(DX)
            M = np.median(np.abs(DX - m))
            Z = (DX - m) / M
            Z[0] = Z[-1] = delta + 1

            # select spikes
            select = abs(Z) >= delta

            # replace spikes by average
            A = np.zeros_like(X)
            for i in [j for j, x in enumerate(select) if x]:
                indexes = np.arange(max(0, i - s), min(len(X), i + s))
                indexes = [index for index in indexes if not select[index]]
                if indexes != []:
                    A[i] = np.mean(X[indexes])
                else:
                    A[i] = X[i]

            # makes change in X
            X[select] -= (X - A)[select]

        new.data[k] = X
    new.history = f"despiked with method={method}, size={size}, delta={delta}"

    return new
