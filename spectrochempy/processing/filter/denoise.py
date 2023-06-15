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


def despike(dataset, **kwargs):
    """
    Despike the data using various algorithm.

    Parameters
    ----------
    dataset
    kwargs

    Returns
    -------

    """
