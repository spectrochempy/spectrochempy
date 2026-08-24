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
cross-validation, pipelines) where feature-wise statistics must be
learned once and reused across multiple datasets.  Sample-local
operations such as SNV and normalization along ``x`` compute their
statistics from the dataset passed to ``transform()``.

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

    _learned_attributes = ()

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
        self._invalidate_fitted_state()
        try:
            self._fit(dataset)
        except Exception:
            self._invalidate_fitted_state()
            raise
        else:
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
        valid = self.get_params()
        invalid = [key for key in params if key not in valid]
        if invalid:
            key = invalid[0]
            valid_names = ", ".join(sorted(valid))
            raise SpectroChemPyError(
                f"Invalid parameter '{key}' for {self.__class__.__name__}. "
                f"Valid parameters: {valid_names}."
            )

        changed = any(
            not self._parameter_values_equal(valid[key], value)
            for key, value in params.items()
        )
        for key, value in params.items():
            setattr(self, key, value)
        if changed:
            self._invalidate_fitted_state()
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        params = self.get_params()
        if not params:
            return f"{cls}()"
        items = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{cls}({items})"

    def _set_data(self, new, data):
        r"""Assign transformed data, coercing MaskedArray → plain ndarray."""
        new._data = np.asarray(data)

    def _invalidate_fitted_state(self):
        r"""Mark the transformer as unfitted and clear declared learned state."""
        self._fitted = False
        for name in self._learned_attributes:
            if hasattr(self, name):
                delattr(self, name)

    @classmethod
    def _parameter_values_equal(cls, old, new):
        r"""Return whether two constructor parameter values are effectively equal."""
        if old is new:
            return True
        if old is None or new is None:
            return False
        if cls._is_spectrochempy_object(old) or cls._is_spectrochempy_object(new):
            return False
        if isinstance(old, np.ma.MaskedArray) or isinstance(new, np.ma.MaskedArray):
            try:
                return bool(np.ma.allequal(old, new))
            except (TypeError, ValueError):
                return False
        if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
            try:
                return bool(np.array_equal(old, new, equal_nan=True))
            except (TypeError, ValueError):
                return False
        try:
            return bool(old == new)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_spectrochempy_object(value):
        return hasattr(value, "masked_data") or hasattr(value, "coordset")

    def _record_fit_compatibility(self, dataset, axis, dim_name):
        r"""Store the dataset signature required to reuse learned statistics."""
        shape = dataset.masked_data.shape
        self._fit_signature_ = {
            "ndim": len(shape),
            "dims": tuple(str(name) for name in dataset.dims),
            "axis": axis,
            "dim_name": str(dim_name),
            "shape": tuple(shape),
            "data_units": dataset.units,
            "coords": self._compatibility_coord_signatures(dataset, axis),
        }

    def _check_fit_compatibility(self, dataset, axis, action):
        r"""Validate that *dataset* can reuse statistics learned during fit()."""
        signature = self._fit_signature_
        cls = self.__class__.__name__
        current_shape = dataset.masked_data.shape
        current_dims = tuple(str(name) for name in dataset.dims)

        if len(current_shape) != signature["ndim"]:
            self._raise_incompatible(action, "number of dimensions changed", cls)

        if current_dims != signature["dims"]:
            self._raise_incompatible(action, "dimension order changed", cls)

        if axis != signature["axis"]:
            self._raise_incompatible(action, "preprocessing axis changed", cls)

        current_dim_name = current_dims[axis]
        if current_dim_name != signature["dim_name"]:
            self._raise_incompatible(action, "preprocessing dimension changed", cls)

        if not self._same_units(dataset.units, signature["data_units"]):
            self._raise_incompatible(action, "data units changed", cls)

        for current_axis, current_size in enumerate(current_shape):
            if current_axis == axis:
                continue
            dim_name = current_dims[current_axis]
            expected_size = signature["shape"][current_axis]
            if current_size != expected_size:
                self._raise_incompatible(
                    action,
                    f"non-reduced dimension '{dim_name}' length changed",
                    cls,
                )

            expected = signature["coords"][dim_name]
            current = self._feature_coord_signature(dataset, dim_name)
            self._check_coord_compatibility(expected, current, dim_name, action, cls)

    @classmethod
    def _compatibility_coord_signatures(cls, dataset, axis):
        signatures = {}
        for current_axis, dim_name in enumerate(dataset.dims):
            if current_axis == axis:
                continue
            name = str(dim_name)
            signatures[name] = cls._feature_coord_signature(dataset, name)
        return signatures

    @staticmethod
    def _feature_coord_signature(dataset, dim_name):
        coord = dataset.coord(dim_name)
        if coord is None or getattr(coord, "data", None) is None:
            return {"present": False, "units": None, "data": None}
        return {
            "present": True,
            "units": coord.units,
            "data": np.array(coord.data, copy=True),
        }

    @classmethod
    def _check_coord_compatibility(cls, expected, current, dim_name, action, owner):
        if current["present"] != expected["present"]:
            cls._raise_incompatible(
                action,
                f"coordinate presence changed for dimension '{dim_name}'",
                owner,
            )

        if not expected["present"]:
            return

        if (current["units"] is None) != (expected["units"] is None):
            cls._raise_incompatible(
                action,
                f"coordinate units changed for dimension '{dim_name}'",
                owner,
            )

        data = np.array(current["data"], copy=True)
        if current["units"] is not None and current["units"] != expected["units"]:
            if current["units"].dimensionality != expected["units"].dimensionality:
                cls._raise_incompatible(
                    action,
                    f"coordinate units are incompatible for dimension '{dim_name}'",
                    owner,
                )
            data = (data * current["units"]).to(expected["units"]).magnitude

        if data.shape != expected["data"].shape or not np.allclose(
            data,
            expected["data"],
            rtol=1.0e-12,
            atol=1.0e-15,
            equal_nan=True,
        ):
            cls._raise_incompatible(
                action,
                f"coordinates changed for dimension '{dim_name}'",
                owner,
            )

    @staticmethod
    def _same_units(current, expected):
        if current is None or expected is None:
            return current is expected
        return current == expected

    @staticmethod
    def _raise_incompatible(action, reason, owner):
        raise SpectroChemPyError(
            f"{owner} {action} dataset is incompatible with fit(): {reason}."
        )

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

    _learned_attributes = ("mean_", "_dim_name", "_fit_signature_")

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        self._record_fit_compatibility(dataset, axis, self._dim_name)
        self.mean_ = np.ma.mean(dataset.masked_data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "transform")
        self._set_data(new, dataset.masked_data - self.mean_)
        new.history = f"CenterTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")
        self._set_data(new, dataset.masked_data + self.mean_)
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

    _learned_attributes = ("mean_", "std_", "_dim_name", "_fit_signature_")

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        self._record_fit_compatibility(dataset, axis, self._dim_name)
        data = dataset.masked_data
        self.mean_ = np.ma.mean(data, axis=axis, keepdims=True)
        self.std_ = np.ma.std(data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "transform")
        data = dataset.masked_data
        std_safe = np.where(self.std_ == 0, 1, self.std_)
        self._set_data(new, (data - self.mean_) / std_safe)
        new.history = f"AutoscaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")
        data = dataset.masked_data
        std_safe = np.where(self.std_ == 0, 1, self.std_)
        self._set_data(new, data * std_safe + self.mean_)
        new.history = (
            f"AutoscaleTransformer inverse applied on dimension {self._dim_name}"
        )
        return new


class SNVTransformer(AutoscaleTransformer):
    r"""
    Standard Normal Variate (SNV) transformer.

    Each observation (spectrum) is mean-centered and scaled to unit
    variance individually.  ``fit()`` validates the preprocessing axis,
    while ``transform()`` computes the per-spectrum statistics on the
    dataset being transformed.

    This transformer hard-codes ``dim='x'`` and provides a descriptive
    name for the common NIR preprocessing step.

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

    _learned_attributes = ("_dim_name",)

    def __init__(self):
        super().__init__(dim="x")

    def _fit(self, dataset):
        _, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, dim_name = dataset.get_axis(self.dim)
        if str(dim_name) != self._dim_name:
            raise SpectroChemPyError(
                "SNVTransformer transform dimension is incompatible with fit(). "
                "Refit the transformer after changing dimensions."
            )
        data = dataset.masked_data
        mean = np.ma.mean(data, axis=axis, keepdims=True)
        std = np.ma.std(data, axis=axis, keepdims=True)
        std_safe = np.where(std == 0, 1, std)
        self._set_data(new, (data - mean) / std_safe)
        new.history = "SNVTransformer applied"
        return new

    def _inverse_transform(self, dataset):
        raise SpectroChemPyError(
            "SNVTransformer inverse_transform is not supported because SNV "
            "computes sample-local statistics during transform()."
        )


class NormalizeTransformer(BasePreprocessor):
    r"""
    Normalization transformer.

    Learns the normalization factor along a reusable feature dimension
    during ``fit()`` and applies it during ``transform()``.  For the
    spectral dimension ``dim='x'``, normalization is sample-local:
    ``fit()`` validates the mode and ``transform()`` computes each
    observation's normalization factor on the dataset being transformed.

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

    _learned_attributes = (
        "norm_",
        "dmin_",
        "dmax_",
        "range_",
        "_dim_name",
        "_sample_local_",
        "_fit_signature_",
    )

    def __init__(self, method="max", dim="x"):
        super().__init__(dim=dim)
        self.method = method

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        data = dataset.masked_data
        self._sample_local_ = self._dim_name == "x"

        if self.method not in ("max", "sum", "vector", "minmax"):
            raise SpectroChemPyError(
                f"Unknown normalization method '{self.method}'. "
                f"Choose from 'max', 'sum', 'vector', 'minmax'."
            )

        if self._sample_local_:
            return

        self._record_fit_compatibility(dataset, axis, self._dim_name)
        params = self._normalization_parameters(data, axis)
        for name, value in params.items():
            setattr(self, name, value)

    def _normalization_parameters(self, data, axis):
        if self.method == "max":
            norm = np.ma.max(np.ma.abs(data), axis=axis, keepdims=True)
            return {"norm_": np.where(norm == 0, 1, norm)}

        if self.method == "sum":
            norm = np.ma.sum(np.ma.abs(data), axis=axis, keepdims=True)
            return {"norm_": np.where(norm == 0, 1, norm)}

        if self.method == "vector":
            norm = np.sqrt(np.ma.sum(data**2, axis=axis, keepdims=True))
            return {"norm_": np.where(norm == 0, 1, norm)}

        dmin = np.ma.min(data, axis=axis, keepdims=True)
        dmax = np.ma.max(data, axis=axis, keepdims=True)
        drange = dmax - dmin
        return {
            "dmin_": dmin,
            "dmax_": dmax,
            "range_": np.where(drange == 0, 1, drange),
        }

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data
        axis, dim_name = dataset.get_axis(self.dim)
        dim_name = str(dim_name)

        if self._sample_local_:
            if dim_name != self._dim_name:
                raise SpectroChemPyError(
                    "NormalizeTransformer transform dimension is incompatible with "
                    "fit(). Refit the transformer after changing dimensions."
                )
            params = self._normalization_parameters(data, axis)
            norm = params.get("norm_")
            dmin = params.get("dmin_")
            drange = params.get("range_")
        else:
            self._check_fit_compatibility(dataset, axis, "transform")
            norm = getattr(self, "norm_", None)
            dmin = getattr(self, "dmin_", None)
            drange = getattr(self, "range_", None)

        if self.method in ("max", "sum", "vector"):
            self._set_data(new, data / norm)
        elif self.method == "minmax":
            self._set_data(new, (data - dmin) / drange)

        new.history = (
            f"NormalizeTransformer ({self.method}) applied on dimension "
            f"{self._dim_name}"
        )
        return new

    def _inverse_transform(self, dataset):
        if self._sample_local_:
            raise SpectroChemPyError(
                "NormalizeTransformer inverse_transform is not supported for "
                "sample-local normalization because normalization factors are "
                "computed during transform()."
            )

        new = dataset.copy()
        data = dataset.masked_data
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")

        if self.method in ("max", "sum", "vector"):
            self._set_data(new, data * self.norm_)
        elif self.method == "minmax":
            self._set_data(new, data * self.range_ + self.dmin_)

        new.history = (
            f"NormalizeTransformer ({self.method}) inverse applied on dimension "
            f"{self._dim_name}"
        )
        return new


class MSCTransformer(BasePreprocessor):
    r"""
    Multiplicative Scatter Correction (MSC) transformer.

    Learns a reusable reference spectrum during ``fit()``.  During
    ``transform()``, each observation in the dataset being transformed is
    locally regressed against that reference and corrected with
    :math:`(x - a) / b`.

    ``inverse_transform()`` is not supported.  The regression coefficients
    are local to each transformed dataset and are not safe reusable state.

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
        Intercepts of the per-observation regressions for the dataset
        passed to ``fit()``. These are diagnostics, not reusable
        transform state.
    b_ : `~numpy.ndarray`
        Slopes of the per-observation regressions for the dataset passed
        to ``fit()``. These are diagnostics, not reusable transform state.

    Raises
    ------
    SpectroChemPyError
        If the dataset is not 2-D, the reference or transform spectral
        geometry is incompatible, a spectrum has too few valid paired
        points, the effective reference is constant, the local slope is
        zero, or ``inverse_transform()`` is requested.

    Examples
    --------
    >>> scaler = scp.MSCTransformer()
    >>> scaler.fit(train)
    >>> test_msc = scaler.transform(test)

    See Also
    --------
    msc : Procedural MSC function.

    """

    _learned_attributes = (
        "reference_",
        "a_",
        "b_",
        "_dim_name",
        "_spectral_axis_",
        "_spectral_dim_name_",
        "_spectral_size_",
        "_spectral_coord_",
    )

    def __init__(self, reference=None, dim="y"):
        super().__init__(dim=dim)
        self.reference = reference

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        data = dataset.masked_data

        if data.ndim != 2:
            raise SpectroChemPyError(
                "MSCTransformer currently supports only 2-D datasets."
            )

        spectral_axis = 1 if axis == 0 else 0
        self._spectral_axis_ = spectral_axis
        self._spectral_dim_name_ = str(dataset.dims[spectral_axis])
        self._spectral_size_ = data.shape[spectral_axis]
        self._spectral_coord_ = self._coord_signature(dataset, self._spectral_dim_name_)

        if self.reference is None:
            ref = np.ma.mean(data, axis=axis)
        else:
            if hasattr(self.reference, "masked_data"):
                ref = self.reference.masked_data
                self._check_reference_coord_compatibility(dataset, self.reference)
            else:
                ref = np.ma.masked_invalid(np.ma.asarray(self.reference))
            if ref.ndim != 1:
                raise SpectroChemPyError("MSC reference must be a 1-D spectrum.")
            if ref.size != data.shape[spectral_axis]:
                raise SpectroChemPyError(
                    f"MSC reference size ({ref.size}) does not match "
                    f"dataset spectral size ({data.shape[spectral_axis]})."
                )

        self.reference_ = np.ma.array(ref, copy=True)
        self.a_, self.b_ = self._msc_coefficients(data, self.reference_, spectral_axis)

    @staticmethod
    def _coord_signature(dataset, dim_name):
        coord = dataset.coord(dim_name)
        if coord is None:
            return None
        return {
            "data": np.array(coord.data, copy=True),
            "units": coord.units,
        }

    def _check_reference_coord_compatibility(self, dataset, reference):
        ref_dim_name = str(reference.dims[0])
        ref_coord = reference.coord(ref_dim_name)
        if ref_coord is None:
            return
        spectral_coord = dataset.coord(self._spectral_dim_name_)
        if spectral_coord is None:
            return
        if ref_coord.units != spectral_coord.units:
            raise SpectroChemPyError(
                "MSC reference coordinates are incompatible with the dataset "
                "spectral coordinates: units differ."
            )
        if not np.array_equal(ref_coord.data, spectral_coord.data):
            raise SpectroChemPyError(
                "MSC reference coordinates are incompatible with the dataset "
                "spectral coordinates."
            )

    def _check_spectral_compatibility(self, dataset):
        axis, dim_name = dataset.get_axis(self.dim)
        dim_name = str(dim_name)
        if dim_name != self._dim_name:
            raise SpectroChemPyError(
                "MSCTransformer transform dimension is incompatible with fit(). "
                "Refit the transformer after changing dimensions."
            )

        spectral_axis = 1 if axis == 0 else 0
        spectral_dim_name = str(dataset.dims[spectral_axis])
        if spectral_dim_name != self._spectral_dim_name_:
            raise SpectroChemPyError(
                "MSCTransformer spectral dimension is incompatible with fit()."
            )
        if dataset.masked_data.shape[spectral_axis] != self._spectral_size_:
            raise SpectroChemPyError(
                "MSCTransformer spectral size is incompatible with fit()."
            )

        coord_signature = self._coord_signature(dataset, spectral_dim_name)
        if self._spectral_coord_ is not None and coord_signature is not None:
            if coord_signature["units"] != self._spectral_coord_["units"]:
                raise SpectroChemPyError(
                    "MSCTransformer spectral coordinates are incompatible "
                    "with fit(): units differ."
                )
            if not np.array_equal(
                coord_signature["data"], self._spectral_coord_["data"]
            ):
                raise SpectroChemPyError(
                    "MSCTransformer spectral coordinates are incompatible "
                    "with fit(): coordinate order changed."
                )

        return spectral_axis

    def _msc_coefficients(self, data, ref, spectral_axis):
        data = np.ma.asarray(data)
        ref = np.ma.asarray(ref)
        ref_shape = [1, 1]
        ref_shape[spectral_axis] = -1
        ref_b = ref.reshape(ref_shape)

        data_mask = np.ma.getmaskarray(data)
        ref_data = np.broadcast_to(np.ma.getdata(ref_b), data.shape)
        ref_mask = np.broadcast_to(np.ma.getmaskarray(ref_b), data.shape)
        combined_mask = data_mask | ref_mask

        x = np.ma.array(np.ma.getdata(data), mask=combined_mask)
        r = np.ma.array(ref_data, mask=combined_mask)

        n = np.ma.count(x, axis=spectral_axis, keepdims=True)
        if np.any(n < 2):
            raise SpectroChemPyError(
                "MSC requires at least two valid spectral points per spectrum."
            )

        ref_range = np.ma.max(r, axis=spectral_axis, keepdims=True) - np.ma.min(
            r, axis=spectral_axis, keepdims=True
        )
        ref_range_data = np.ma.asarray(ref_range).filled(np.nan)
        if np.any((ref_range_data == 0.0) | ~np.isfinite(ref_range_data)):
            raise SpectroChemPyError(
                "MSC denominator is zero; reference spectrum is constant over "
                "the valid points."
            )

        mean_ref = np.ma.mean(r, axis=spectral_axis, keepdims=True)
        mean_x = np.ma.mean(x, axis=spectral_axis, keepdims=True)
        r_centered = r - mean_ref
        x_centered = x - mean_x
        den = np.ma.sum(r_centered**2, axis=spectral_axis, keepdims=True)

        den_data = np.ma.asarray(den).filled(np.nan)
        if np.any((den_data == 0.0) | ~np.isfinite(den_data)):
            raise SpectroChemPyError(
                "MSC denominator is zero; reference spectrum is constant over "
                "the valid points."
            )

        covariance = np.ma.sum(
            x_centered * r_centered, axis=spectral_axis, keepdims=True
        )
        covariance_data = np.ma.asarray(covariance).filled(np.nan)
        if np.any(~np.isfinite(covariance_data)):
            raise SpectroChemPyError(
                "MSC covariance is not finite; the spectrum cannot be corrected "
                "against the reference."
            )

        b = covariance / den
        b_data = np.ma.asarray(b).filled(np.nan)
        if np.any((b_data == 0.0) | ~np.isfinite(b_data)):
            raise SpectroChemPyError(
                "MSC slope is zero; the spectrum cannot be corrected against "
                "the reference."
            )
        a = mean_x - b * mean_ref
        return a, b

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if data.ndim != 2:
            raise SpectroChemPyError(
                "MSCTransformer currently supports only 2-D datasets."
            )

        spectral_axis = self._check_spectral_compatibility(dataset)

        ref = self.reference_
        a, b = self._msc_coefficients(data, ref, spectral_axis)
        self._set_data(new, (data - a) / b)
        new.history = f"MSCTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        raise SpectroChemPyError(
            "MSCTransformer inverse_transform is not supported because MSC "
            "regression coefficients are local to each transformed dataset."
        )


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

    _learned_attributes = ("mean_", "std_", "_dim_name", "_fit_signature_")

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        self._record_fit_compatibility(dataset, axis, self._dim_name)
        data = dataset.masked_data
        self.mean_ = np.ma.mean(data, axis=axis, keepdims=True)
        self.std_ = np.ma.std(data, axis=axis, keepdims=True)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "transform")
        data = dataset.masked_data

        std_safe = np.where(self.std_ == 0, 1, self.std_)
        self._set_data(new, (data - self.mean_) / np.sqrt(std_safe))
        new.history = f"ParetoScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")
        data = dataset.masked_data

        std_safe = np.where(self.std_ == 0, 1, self.std_)
        self._set_data(new, data * np.sqrt(std_safe) + self.mean_)
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

    _learned_attributes = ("dmin_", "dmax_", "range_", "_dim_name", "_fit_signature_")

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        self._record_fit_compatibility(dataset, axis, self._dim_name)
        data = dataset.masked_data

        self.dmin_ = np.ma.min(data, axis=axis, keepdims=True)
        self.dmax_ = np.ma.max(data, axis=axis, keepdims=True)
        self.range_ = self.dmax_ - self.dmin_
        self.range_ = np.where(self.range_ == 0, 1, self.range_)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "transform")
        data = dataset.masked_data

        self._set_data(new, data / self.range_)
        new.history = f"RangeScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")
        data = dataset.masked_data

        self._set_data(new, data * self.range_)
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

    _learned_attributes = ("median_", "mad_", "_dim_name", "_fit_signature_")

    def __init__(self, dim="y"):
        super().__init__(dim=dim)

    def _fit(self, dataset):
        axis, self._dim_name = dataset.get_axis(self.dim)
        self._dim_name = str(self._dim_name)
        self._record_fit_compatibility(dataset, axis, self._dim_name)
        data = dataset.masked_data

        self.median_ = np.ma.median(data, axis=axis, keepdims=True)
        mad = np.ma.median(np.ma.abs(data - self.median_), axis=axis, keepdims=True)
        self.mad_ = mad * 1.4826
        self.mad_ = np.where(self.mad_ == 0, 1, self.mad_)

    def _transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "transform")
        data = dataset.masked_data

        self._set_data(new, (data - self.median_) / self.mad_)
        new.history = f"RobustScaleTransformer applied on dimension {self._dim_name}"
        return new

    def _inverse_transform(self, dataset):
        new = dataset.copy()
        axis, _ = dataset.get_axis(self.dim)
        self._check_fit_compatibility(dataset, axis, "inverse_transform")
        data = dataset.masked_data

        self._set_data(new, data * self.mad_ + self.median_)
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

    _learned_attributes = ()

    def __init__(self, method="log1p", eps=1e-10):
        super().__init__(dim=None)
        self.method = method
        self.eps = eps

    def _fit(self, dataset):
        if self.method not in ("log1p", "log"):
            raise SpectroChemPyError(
                f"Unknown LogTransformer method '{self.method}'. "
                f"Choose from 'log1p' or 'log'."
            )

    def _transform(self, dataset):
        new = dataset.copy()
        data = dataset.masked_data

        if self.method == "log1p":
            self._set_data(new, np.log1p(data))
            new.history = "LogTransformer (log1p) applied"
        elif self.method == "log":
            if np.any(data <= 0):
                data = data + self.eps
            self._set_data(new, np.log(data))
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
            self._set_data(new, np.expm1(data))
            new.history = "LogTransformer (log1p) inverse applied"
        elif self.method == "log":
            self._set_data(new, np.exp(data))
            new.history = "LogTransformer (log) inverse applied"
        else:
            raise SpectroChemPyError(
                f"Unknown LogTransformer method '{self.method}'. "
                f"Choose from 'log1p' or 'log'."
            )
        return new
