"""
Trait types module for scientific data objects.

This module provides trait types for numpy arrays, pandas dataframes, pandas series,
xarray datasets and xarray dataarrays.

Notes
-----
Imported from https://github.com/jupyter-widgets/traittypes
See LICENSES in the root folder.

"""

import inspect
import warnings

from traitlets import TraitError
from traitlets import TraitType
from traitlets import Undefined

from spectrochempy.extern.traittypes_utils import Sentinel


class _DelayedImportError:
    def __init__(self, package_name):
        self.package_name = package_name

    def __getattribute__(self, name):
        package_name = super().__getattribute__("package_name")
        raise RuntimeError(f"Missing dependency: {package_name}")


try:
    import numpy as np
except ImportError:
    np = _DelayedImportError("numpy")


Empty = Sentinel(
    "Empty",
    "traittypes",
    """
Used in traittypes to specify that the default value should
be an empty dataset
""",
)


class SciType(TraitType):
    """
    Base trait type for scientific data objects.

    A base class for numpy arrays, pandas dataframes, pandas series,
    xarray datasets and xarray dataarrays.

    Attributes
    ----------
    validators : list
        List of validator functions to apply to values.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validators = []

    def valid(self, *validators):
        """
        Register new trait validators.

        Parameters
        ----------
        *validators : callable
            Functions that take two arguments:
            - The trait instance
            - The proposed value

            Validators return the (potentially modified) value, which is either
            assigned to the HasTraits attribute or input into the next validator.

        Returns
        -------
        SciType
            Self, to allow method chaining.

        Examples
        --------
        Test with a shape constraint:

        >>> def shape(*dimensions):
        ...     def validator(trait, value):
        ...         if value.shape != dimensions:
        ...             raise TraitError(
        ...                 f'Expected shape {dimensions}, got shape {value.shape}'
        ...             )
        ...         return value
        ...     return validator
        >>> class Foo(HasTraits):
        ...     bar = Array(np.identity(2)).valid(shape(2, 2))
        >>> foo = Foo()
        >>> foo.bar = [1, 2]  # Should raise a TraitError

        """
        self.validators.extend(validators)
        return self

    def validate(self, obj, value):
        """Validate the value against registered validators."""
        try:
            for validator in self.validators:
                value = validator(self, value)
            return value
        except (ValueError, TypeError) as e:
            raise TraitError(e) from None


class Array(SciType):
    """
    Numpy array trait type.

    Parameters
    ----------
    default_value : array-like, optional
        The default value for the trait. If Empty, defaults to np.array(0).
    allow_none : bool, optional
        Whether to allow None as a valid value, by default False.
    dtype : numpy.dtype, optional
        The dtype to enforce on values.

    """

    info_text = "a numpy array"
    dtype = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super().validate(obj, value)
        try:
            r = np.asarray(value, dtype=self.dtype)
            if isinstance(value, np.ndarray) and r is not value:
                warnings.warn(
                    f'Given trait value dtype "{np.dtype(value.dtype).name}" does not match required type "{np.dtype(self.dtype).name}". '
                    "A coerced copy has been created.",
                    stacklevel=2,
                )
            value = r
        except (ValueError, TypeError) as e:
            raise TraitError(e) from None
        return super().validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if not np.array_equal(old_value, new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        self.dtype = dtype
        if default_value is Empty:
            default_value = np.array(0, dtype=self.dtype)
        elif default_value is not None and default_value is not Undefined:
            default_value = np.asarray(default_value, dtype=self.dtype)
        super().__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        return np.copy(self.default_value)


class PandasType(SciType):
    """A pandas dataframe or series trait type."""

    info_text = "a pandas dataframe or series"

    klass = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super().validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise TraitError(e) from None
        return super().validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (
            (old_value is None and new_value is not None)
            or (old_value is Undefined and new_value is not Undefined)
            or not old_value.equals(new_value)
        ):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Empty, allow_none=False, klass=None, **kwargs):
        if klass is None:
            klass = self.klass
        if (klass is not None) and inspect.isclass(klass):
            self.klass = klass
        else:
            raise TraitError(f"The klass attribute must be a class not: {klass!r}")
        if default_value is Empty:
            default_value = klass()
        elif default_value is not None and default_value is not Undefined:
            default_value = klass(default_value)
        super().__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        return self.default_value.copy()


class DataFrame(PandasType):
    """A pandas dataframe trait type."""

    info_text = "a pandas dataframe"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import pandas as pd  # type: ignore

            kwargs["klass"] = pd.DataFrame
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


class Series(PandasType):
    """A pandas series trait type."""

    info_text = "a pandas series"
    dtype = None

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import pandas as pd  # type: ignore

            kwargs["klass"] = pd.Series
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )
        self.dtype = dtype


class XarrayType(SciType):
    """An xarray dataset or dataarray trait type."""

    info_text = "an xarray dataset or dataarray"

    klass = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super().validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise TraitError(e) from None
        return super().validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (
            (old_value is None and new_value is not None)
            or (old_value is Undefined and new_value is not Undefined)
            or not old_value.equals(new_value)
        ):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Empty, allow_none=False, klass=None, **kwargs):
        if klass is None:
            klass = self.klass
        if (klass is not None) and inspect.isclass(klass):
            self.klass = klass
        else:
            raise TraitError(f"The klass attribute must be a class not: {klass!r}")
        if default_value is Empty:
            default_value = klass()
        elif default_value is not None and default_value is not Undefined:
            default_value = klass(default_value)
        super().__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        return self.default_value.copy()


class Dataset(XarrayType):
    """An xarray dataset trait type."""

    info_text = "an xarray dataset"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import xarray as xr  # type: ignore

            kwargs["klass"] = xr.Dataset
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


class DataArray(XarrayType):
    """An xarray dataarray trait type."""

    info_text = "an xarray dataarray"
    dtype = None

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import xarray as xr  # type: ignore

            kwargs["klass"] = xr.DataArray
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )
        self.dtype = dtype
