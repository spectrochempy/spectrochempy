# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Stateful preprocessing transformers.

These classes implement a scikit-learn-style ``fit()`` / ``transform()`` /
``fit_transform()`` / ``inverse_transform()`` API for preprocessing
operations that learn parameters from data.

They complement the procedural functions in
:mod:`spectrochempy.processing.transformation.preprocessing` and are
intended for machine-learning workflows (train/test splits,
cross-validation, pipelines) where statistics must be learned once and
reused across multiple datasets.

"""

__all__ = [
    "BasePreprocessor",
    "CenterTransformer",
    "AutoscaleTransformer",
    "SNVTransformer",
]

__dataset_methods__ = [
    "CenterTransformer",
    "AutoscaleTransformer",
    "SNVTransformer",
]

import numpy as np

from spectrochempy.utils.exceptions import SpectroChemPyError


class BasePreprocessor:
    r"""
    Base class for stateful preprocessing transformers.

    Provides the common ``fit()`` / ``transform()`` / ``fit_transform()``
    lifecycle and tracks whether the transformer has already been fitted.

    Subclasses must implement:

    - ``_fit(dataset)`` — compute and store learned parameters.
    - ``_transform(dataset)`` — apply the learned transformation.
    - ``_inverse_transform(dataset)`` — reverse the transformation (optional).

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which statistics are computed and applied.

    Examples
    --------
    >>> scaler = scp.AutoscaleTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_scaled = scaler.transform(test)

    """

    def __init__(self, dim="y"):
        self.dim = dim
        self._fitted = False

    def fit(self, dataset):
        r"""
        Learn parameters from *dataset*.

        Parameters
        ----------
        dataset : `NDDataset`
            Training data.

        Returns
        -------
        self
            The fitted instance.

        """
        self._fit(dataset)
        self._fitted = True
        return self

    def transform(self, dataset):
        r"""
        Apply the learned transformation to *dataset*.

        Parameters
        ----------
        dataset : `NDDataset`
            Data to transform.

        Returns
        -------
        `NDDataset`
            Transformed dataset.

        Raises
        ------
        SpectroChemPyError
            If ``fit()`` has not been called first.

        """
        if not self._fitted:
            raise SpectroChemPyError(
                "This transformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        return self._transform(dataset)

    def fit_transform(self, dataset):
        r"""
        Fit to *dataset*, then transform it.

        Equivalent to ``self.fit(dataset).transform(dataset)`` but
        avoids an intermediate copy when possible.

        Parameters
        ----------
        dataset : `NDDataset`
            Training data.

        Returns
        -------
        `NDDataset`
            Transformed dataset.

        """
        self.fit(dataset)
        return self._transform(dataset)

    def inverse_transform(self, dataset):
        r"""
        Reverse the learned transformation on *dataset*.

        Parameters
        ----------
        dataset : `NDDataset`
            Data to invert.

        Returns
        -------
        `NDDataset`
            Dataset in the original space.

        Raises
        ------
        SpectroChemPyError
            If ``fit()`` has not been called first.

        """
        if not self._fitted:
            raise SpectroChemPyError(
                "This transformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        return self._inverse_transform(dataset)

    def _fit(self, dataset):
        raise NotImplementedError("Subclasses must implement _fit().")

    def _transform(self, dataset):
        raise NotImplementedError("Subclasses must implement _transform().")

    def _inverse_transform(self, dataset):
        raise NotImplementedError("Subclasses must implement _inverse_transform().")


class CenterTransformer(BasePreprocessor):
    r"""
    Mean-centering transformer.

    Learns the mean along a dimension during ``fit()`` and subtracts it
    during ``transform()``.

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the mean is computed.

    Attributes
    ----------
    mean_ : `~numpy.ndarray`
        Learned mean, with shape compatible for broadcasting along ``dim``.

    Examples
    --------
    >>> scaler = scp.CenterTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_centered = scaler.transform(test)
    >>> train_restored = scaler.inverse_transform(test_centered)

    See Also
    --------
    center : Procedural mean-centering function.

    """

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self.mean_ = np.ma.mean(dataset.masked_data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        new._data = dataset.masked_data - self.mean_
        new.history = f"CenterTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        new._data = dataset.masked_data + self.mean_
        new.history = f"CenterTransformer inverse applied on dimension {self._dim_name}"
        return new


class AutoscaleTransformer(BasePreprocessor):
    r"""
    Autoscaling (z-score) transformer.

    Learns the mean and standard deviation along a dimension during
    ``fit()`` and applies :math:`(x - \bar{x}) / s` during ``transform()``.

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the mean and standard deviation are computed.

    Attributes
    ----------
    mean_ : `~numpy.ndarray`
        Learned mean.
    std_ : `~numpy.ndarray`
        Learned standard deviation.

    Examples
    --------
    >>> scaler = scp.AutoscaleTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_scaled = scaler.transform(test)
    >>> train_restored = scaler.inverse_transform(test_scaled)

    See Also
    --------
    autoscale : Procedural autoscaling function.

    """

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data
        self.mean_ = np.ma.mean(data, axis=axis, keepdims=True)
        self.std_ = np.ma.std(data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data
        std_safe = np.where(self.std_ == 0, 1, self.std_)
        new._data = (data - self.mean_) / std_safe
        new.history = f"AutoscaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data
        std_safe = np.where(self.std_ == 0, 1, self.std_)
        new._data = data * std_safe + self.mean_
        new.history = (
            f"AutoscaleTransformer inverse applied on dimension {self._dim_name}"
        )
        return new


class SNVTransformer(AutoscaleTransformer):
    r"""
    Standard Normal Variate (SNV) transformer.

    Equivalent to :class:`AutoscaleTransformer` with ``dim='x'``.
    Each observation (spectrum) is mean-centered and scaled to unit
    variance individually.

    This is a thin wrapper that hard-codes ``dim='x'`` and provides a
    more descriptive name for the common NIR preprocessing step.

    Examples
    --------
    >>> scaler = scp.SNVTransformer()
    >>> scaler.fit(train)
    >>> test_snv = scaler.transform(test)

    See Also
    --------
    snv : Procedural SNV function.
    AutoscaleTransformer : General autoscaling transformer.

    """

    def __init__(self):
        super().__init__(dim="x")

    def _transform(self, dataset):
        new = super()._transform(dataset)
        new.history = "SNVTransformer applied"
        return new

    def _inverse_transform(self, dataset):
        new = super()._inverse_transform(dataset)
        new.history = "SNVTransformer inverse applied"
        return new
