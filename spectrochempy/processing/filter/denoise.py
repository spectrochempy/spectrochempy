import numpy as np

from spectrochempy.application import error_, info_, warning_
from spectrochempy.core import get_loglevel

__dataset_methods__ = [
    "denoise",
    "despike",
]
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
        f"[n_observations={dataset.shape[0]}, ratio={ratio*100:.2f}%]"
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


def despike(dataset, size=9, delta=2):
    """
    Remove convex spike from the data using the katsumoto-ozaki procedure.

    The method can be used to remove cosmic ray peaks from a spectrum.

    The present implementation is based on the method is described
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


    Parameters
    ----------
    dataset : `NDDataset` or a ndarray-like object
        Input object.
    size : int, optional, default: 9
        Size of the moving average window.
    delta : float, optional, default: 2
        Set the threshold for the detection of spikes. A spike is detected if its value
        is greater than `delta` times the standard deviation of the difference between
        the original and the smoothed data.

    Returns
    -------
    `NDdataset`
        The despike dataset
    """
    new = dataset.copy()
    s = int((size - 1) / 2)

    for k, X in enumerate(new):
        X = X.squeeze()

        # 1) first step : savgol filter
        A = X.savgol(size=size)

        # 2 and 3) second and third step : detect spike and replace spkike by the moving
        # average and the new data are smoothed again
        diff = X - A
        std = delta * diff.std()

        # spike should have a large variation with respect to the std of the difference
        select = np.array(abs(diff) >= std, dtype=bool)
        select = np.logical_or(select, np.roll(select, 1))
        select = np.logical_or(select, np.roll(select, -1))

        # compute weights
        w = np.ones_like(X)
        w[select] = 0

        # now we must calculate the weighted average, but for efficiency we will compute it
        # only around where spike peaks are, the other part should be unchanged.

        res = np.zeros_like(X)
        W = np.zeros_like(X)
        r = range(-s, s + 1)
        for j in r:
            vj = np.roll(X, j)
            wj = np.roll(w, j)
            res += vj * wj
            W += wj
        if 0 in W:
            error_("May be size or delta is two low")
        A = res / W

        # 4) compare with original to remove spike peaks
        X.data[select] -= (X - A).data[select]
        new[k] = X

    return new
