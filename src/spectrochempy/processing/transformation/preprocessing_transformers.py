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

They also provide ``get_params()`` and ``set_params()`` so that they can
be cloned and composed in pipeline-like workflows that follow
scikit-learn conventions (scikit-learn itself is **not** a dependency).

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
    "NormalizeTransformer",
    "MSCTransformer",
    "ParetoScaleTransformer",
    "RangeScaleTransformer",
    "RobustScaleTransformer",
    "LogTransformer",
]

__dataset_methods__ = [
    "CenterTransformer",
    "AutoscaleTransformer",
    "SNVTransformer",
    "NormalizeTransformer",
    "MSCTransformer",
    "ParetoScaleTransformer",
    "RangeScaleTransformer",
    "RobustScaleTransformer",
    "LogTransformer",
]

import inspect

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

    def get_params(self, deep=True):
        r"""
        Get the constructor parameters of this transformer.

        Parameters
        ----------
        deep : `bool`, optional, default:`True`
            Ignored.  Present for compatibility with scikit-learn conventions.

        Returns
        -------
        `dict`
            Mapping of parameter name -> current value.

        Examples
        --------
        >>> scaler = scp.AutoscaleTransformer(dim="y")
        >>> scaler.get_params()
        {'dim': 'y'}

        """
        sig = inspect.signature(self.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif param.default is not inspect.Parameter.empty:
                params[name] = param.default
        return params

    def set_params(self, **params):
        r"""
        Set constructor parameters on this transformer.

        Returns `self` so that calls can be chained.

        Parameters
        ----------
        **params
            Parameter names and values to update.

        Returns
        -------
        self

        Raises
        ------
        SpectroChemPyError
            If a parameter name does not correspond to a constructor argument.

        Examples
        --------
        >>> scaler = scp.AutoscaleTransformer(dim="y")
        >>> scaler.set_params(dim="x")
        AutoscaleTransformer(dim='x')

        """
        valid = self.get_params().keys()
        for key, value in params.items():
            if key not in valid:
                raise SpectroChemPyError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}. "
                    f"Valid parameters: {', '.join(sorted(valid))}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        params = self.get_params()
        if not params:
            return f"{cls}()"
        items = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{cls}({items})"

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


class NormalizeTransformer(BasePreprocessor):
    r"""
    Normalization transformer.

    Learns the normalization factor along a dimension during ``fit()``
    and applies it during ``transform()``.

    Parameters
    ----------
    method : `str`, optional, default:'max'
        Normalization method:

        * ``'max'``     — divide by the maximum absolute value.
        * ``'sum'``     — divide by the sum of absolute values.
        * ``'vector'``  — divide by the Euclidean (L2) norm.
        * ``'minmax'``  — scale linearly to the range ``[0, 1]``.

    dim : `str` or `int`, optional, default:'x'
        Dimension along which the normalization is computed.

    Attributes
    ----------
    norm_ : `~numpy.ndarray`
        Learned norm (for ``'max'``, ``'sum'``, ``'vector'``).
    dmin_ : `~numpy.ndarray`
        Learned minimum (for ``'minmax'``).
    dmax_ : `~numpy.ndarray`
        Learned maximum (for ``'minmax'``).
    range_ : `~numpy.ndarray`
        Learned range (for ``'minmax'``).

    Examples
    --------
    >>> scaler = scp.NormalizeTransformer(method="max", dim="x")
    >>> scaler.fit(train)
    >>> test_norm = scaler.transform(test)

    See Also
    --------
    normalize : Procedural normalization function.

    """

    def __init__(self, method="max", dim="x"):
        super().__init__(dim=dim)
        self.method = method

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data

        if self.method == "max":
            self.norm_ = np.ma.max(np.ma.abs(data), axis=axis, keepdims=True)
            self.norm_ = np.where(self.norm_ == 0, 1, self.norm_)

        elif self.method == "sum":
            self.norm_ = np.ma.sum(np.ma.abs(data), axis=axis, keepdims=True)
            self.norm_ = np.where(self.norm_ == 0, 1, self.norm_)

        elif self.method == "vector":
            self.norm_ = np.sqrt(np.ma.sum(data**2, axis=axis, keepdims=True))
            self.norm_ = np.where(self.norm_ == 0, 1, self.norm_)

        elif self.method == "minmax":
            self.dmin_ = np.ma.min(data, axis=axis, keepdims=True)
            self.dmax_ = np.ma.max(data, axis=axis, keepdims=True)
            self.range_ = self.dmax_ - self.dmin_
            self.range_ = np.where(self.range_ == 0, 1, self.range_)

        else:
            raise SpectroChemPyError(
                f"Unknown normalization method '{self.method}'. "
                f"Choose from 'max', 'sum', 'vector', 'minmax'."
            )

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if self.method in ("max", "sum", "vector"):
            new._data = data / self.norm_
        elif self.method == "minmax":
            new._data = (data - self.dmin_) / self.range_

        new.history = (
            f"NormalizeTransformer ({self.method}) applied on dimension "
            f"{self._dim_name}"
        )
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if self.method in ("max", "sum", "vector"):
            new._data = data * self.norm_
        elif self.method == "minmax":
            new._data = data * self.range_ + self.dmin_

        new.history = (
            f"NormalizeTransformer ({self.method}) inverse applied on dimension "
            f"{self._dim_name}"
        )
        return new


class MSCTransformer(BasePreprocessor):
    r"""
    Multiplicative Scatter Correction (MSC) transformer.

    Fits a linear regression of each observation against a reference
    spectrum during ``fit()``, then corrects with
    :math:`(x - a) / b` during ``transform()``.

    Parameters
    ----------
    reference : `NDDataset` or array-like, optional
        1-D reference spectrum.  If `None`, the mean spectrum is used.
    dim : `str` or `int`, optional, default:'y'
        Dimension that identifies individual observations (spectra).

    Attributes
    ----------
    reference_ : `~numpy.ndarray`
        Reference spectrum used for fitting.
    a_ : `~numpy.ndarray`
        Intercepts of the per-observation regressions.
    b_ : `~numpy.ndarray`
        Slopes of the per-observation regressions.

    Examples
    --------
    >>> scaler = scp.MSCTransformer()
    >>> scaler.fit(train)
    >>> test_msc = scaler.transform(test)

    See Also
    --------
    msc : Procedural MSC function.

    """

    def __init__(self, reference=None, dim="y"):
        super().__init__(dim=dim)
        self.reference = reference

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data

        if data.ndim != 2:
            raise SpectroChemPyError(
                "MSCTransformer currently supports only 2-D datasets."
            )

        spectral_axis = 1 if axis == 0 else 0

        if self.reference is None:
            ref = np.ma.mean(data, axis=axis)
            ref = np.asarray(ref)
        else:
            if hasattr(self.reference, "masked_data"):
                ref = self.reference.masked_data
            else:
                ref = np.ma.masked_invalid(np.asarray(self.reference))
            if ref.ndim != 1:
                raise SpectroChemPyError("MSC reference must be a 1-D spectrum.")
            if ref.size != data.shape[spectral_axis]:
                raise SpectroChemPyError(
                    f"MSC reference size ({ref.size}) does not match "
                    f"dataset spectral size ({data.shape[spectral_axis]})."
                )

        self.reference_ = ref

        ref_shape = [1, 1]
        ref_shape[spectral_axis] = -1
        ref_b = ref.reshape(ref_shape)

        n = ref.size
        sum_ref = np.ma.sum(ref)
        sum_ref2 = np.ma.sum(ref**2)
        den = n * sum_ref2 - sum_ref**2

        if den == 0:
            raise SpectroChemPyError(
                "MSC denominator is zero; reference spectrum is constant."
            )

        sum_x = np.ma.sum(data, axis=spectral_axis, keepdims=True)
        sum_xref = np.ma.sum(data * ref_b, axis=spectral_axis, keepdims=True)

        self.b_ = (n * sum_xref - sum_ref * sum_x) / den
        self.a_ = (sum_x - self.b_ * sum_ref) / n

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if data.ndim != 2:
            raise SpectroChemPyError(
                "MSCTransformer currently supports only 2-D datasets."
            )

        axis, _ = dataset.get_axis(self.dim)
        spectral_axis = 1 if axis == 0 else 0

        ref = self.reference_
        ref_shape = [1, 1]
        ref_shape[spectral_axis] = -1
        ref_b = ref.reshape(ref_shape)

        n = ref.size
        sum_ref = np.ma.sum(ref)
        sum_ref2 = np.ma.sum(ref**2)
        den = n * sum_ref2 - sum_ref**2

        sum_x = np.ma.sum(data, axis=spectral_axis, keepdims=True)
        sum_xref = np.ma.sum(data * ref_b, axis=spectral_axis, keepdims=True)

        b = (n * sum_xref - sum_ref * sum_x) / den
        a = (sum_x - b * sum_ref) / n

        b_safe = np.where(b == 0, 1, b)
        new._data = (data - a) / b_safe
        new.history = f"MSCTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        new._data = data * self.b_ + self.a_
        new.history = f"MSCTransformer inverse applied on dimension {self._dim_name}"
        return new


class ParetoScaleTransformer(BasePreprocessor):
    r"""
    Pareto scaling transformer.

    Learns the mean and standard deviation along a dimension during
    ``fit()`` and applies :math:`(x - \bar{x}) / \sqrt{s}` during
    ``transform()``.

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the statistics are computed.

    Attributes
    ----------
    mean_ : `~numpy.ndarray`
        Learned mean.
    std_ : `~numpy.ndarray`
        Learned standard deviation.

    Examples
    --------
    >>> scaler = scp.ParetoScaleTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_scaled = scaler.transform(test)

    See Also
    --------
    pareto_scale : Procedural Pareto scaling function.

    """

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data
        self.mean_ = np.ma.mean(data, axis=axis, keepdims=True)
        self.std_ = np.ma.std(data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        std_safe = np.where(self.std_ == 0, 1, self.std_)
        new._data = (data - self.mean_) / np.sqrt(std_safe)
        new.history = f"ParetoScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        std_safe = np.where(self.std_ == 0, 1, self.std_)
        new._data = data * np.sqrt(std_safe) + self.mean_
        new.history = (
            f"ParetoScaleTransformer inverse applied on dimension " f"{self._dim_name}"
        )
        return new


class RangeScaleTransformer(BasePreprocessor):
    r"""
    Range scaling transformer.

    Learns the range (``max - min``) along a dimension during ``fit()``
    and divides by it during ``transform()``.

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the range is computed.

    Attributes
    ----------
    dmin_ : `~numpy.ndarray`
        Learned minimum.
    dmax_ : `~numpy.ndarray`
        Learned maximum.
    range_ : `~numpy.ndarray`
        Learned range.

    Examples
    --------
    >>> scaler = scp.RangeScaleTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_scaled = scaler.transform(test)

    See Also
    --------
    range_scale : Procedural range scaling function.

    """

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data

        self.dmin_ = np.ma.min(data, axis=axis, keepdims=True)
        self.dmax_ = np.ma.max(data, axis=axis, keepdims=True)
        self.range_ = self.dmax_ - self.dmin_
        self.range_ = np.where(self.range_ == 0, 1, self.range_)

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        new._data = data / self.range_
        new.history = f"RangeScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        new._data = data * self.range_
        new.history = (
            f"RangeScaleTransformer inverse applied on dimension " f"{self._dim_name}"
        )
        return new


class RobustScaleTransformer(BasePreprocessor):
    r"""
    Robust scaling transformer.

    Learns the median and median absolute deviation (MAD) along a
    dimension during ``fit()`` and applies
    :math:`(x - \mathrm{median}) / \mathrm{MAD}` during ``transform()``.

    Parameters
    ----------
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the median and MAD are computed.

    Attributes
    ----------
    median_ : `~numpy.ndarray`
        Learned median.
    mad_ : `~numpy.ndarray`
        Learned MAD, scaled by 1.4826 to estimate standard deviation.

    Examples
    --------
    >>> scaler = scp.RobustScaleTransformer(dim="y")
    >>> scaler.fit(train)
    >>> test_scaled = scaler.transform(test)

    See Also
    --------
    robust_scale : Procedural robust scaling function.

    """

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        data = dataset.masked_data

        self.median_ = np.ma.median(data, axis=axis, keepdims=True)
        mad = np.ma.median(np.ma.abs(data - self.median_), axis=axis, keepdims=True)
        self.mad_ = mad * 1.4826
        self.mad_ = np.where(self.mad_ == 0, 1, self.mad_)

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        new._data = (data - self.median_) / self.mad_
        new.history = f"RobustScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        new._data = data * self.mad_ + self.median_
        new.history = (
            f"RobustScaleTransformer inverse applied on dimension " f"{self._dim_name}"
        )
        return new


class LogTransformer(BasePreprocessor):
    r"""
    Logarithmic transform.

    This is a *stateless* transformer: ``fit()`` is a no-op and the
    same transform is applied regardless of the input data.  It is
    provided for API uniformity so that all preprocessing steps can be
    expressed as transformer objects.

    Parameters
    ----------
    method : `str`, optional, default:'log1p'
        Transform to apply:

        * ``'log1p'`` — compute ``log(1 + x)`` (stable for small or zero values).
        * ``'log'``   — compute ``log(x)``.  If the data contain values
          :math:`\le 0`, a small offset ``eps`` is added automatically.

    eps : `float`, optional, default:1e-10
        Offset added when ``method='log'`` and non-positive values are present.

    Examples
    --------
    >>> transformer = scp.LogTransformer(method="log1p")
    >>> nd = transformer.fit_transform(dataset)

    See Also
    --------
    log_transform : Procedural log transform function.

    """

    def __init__(self, method="log1p", eps=1e-10):
        super().__init__(dim=None)
        self.method = method
        self.eps = eps

    def _fit(self, dataset):
        # Stateless — nothing to learn
        pass

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if self.method == "log1p":
            new._data = np.log1p(data)
            new.history = "LogTransformer (log1p) applied"
        elif self.method == "log":
            if np.any(data <= 0):
                data = data + self.eps
            new._data = np.log(data)
            new.history = "LogTransformer (log) applied"
        else:
            raise SpectroChemPyError(
                f"Unknown LogTransformer method '{self.method}'. "
                f"Choose from 'log1p' or 'log'."
            )
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if self.method == "log1p":
            new._data = np.expm1(data)
            new.history = "LogTransformer (log1p) inverse applied"
        elif self.method == "log":
            new._data = np.exp(data)
            new.history = "LogTransformer (log) inverse applied"
        else:
            raise SpectroChemPyError(
                f"Unknown LogTransformer method '{self.method}'. "
                f"Choose from 'log1p' or 'log'."
            )
        return new
