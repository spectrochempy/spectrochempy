# traittypes
# ----------
# imported from
# https://github.com/jupyter-widgets/traittypes
# See LICENSES in the root folder.

import inspect
import warnings

from traitlets import TraitError, TraitType, Undefined

from spectrochempy.extern.traittypes_utils import Sentinel


class _DelayedImportError(object):
    def __init__(self, package_name):
        self.package_name = package_name

    def __getattribute__(self, name):
        package_name = super(_DelayedImportError, self).__getattribute__("package_name")
        raise RuntimeError("Missing dependency: %s" % package_name)


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

    """A base trait type for numpy arrays, pandas dataframes, pandas series, xarray datasets and xarray dataarrays."""

    def __init__(self, **kwargs):
        super(SciType, self).__init__(**kwargs)
        self.validators = []

    def valid(self, *validators):
        """
        Register new trait validators

        Validators are functions that take two arguments.
         - The trait instance
         - The proposed value

        Validators return the (potentially modified) value, which is either
        assigned to the HasTraits attribute or input into the next validator.

        They are evaluated in the order in which they are provided to the `valid`
        function.

        Example
        -------

        .. code:: python

            # Test with a shape constraint
            def shape(*dimensions):
                def validator(trait, value):
                    if value.shape != dimensions:
                        raise TraitError('Expected an of shape %s and got and array with
                         shape %s' % (dimensions, value.shape))
                    else:
                        return value
                return validator

            class Foo(HasTraits):
                bar = Array(np.identity(2)).valid(shape(2, 2))
            foo = Foo()

            foo.bar = [1, 2]  # Should raise a TraitError
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
            raise TraitError(e)


class Array(SciType):

    """A numpy array trait type."""

    info_text = "a numpy array"
    dtype = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super(Array, self).validate(obj, value)
        try:
            r = np.asarray(value, dtype=self.dtype)
            if isinstance(value, np.ndarray) and r is not value:
                warnings.warn(
                    'Given trait value dtype "%s" does not match required type "%s". '
                    "A coerced copy has been created."
                    % (np.dtype(value.dtype).name, np.dtype(self.dtype).name)
                )
            value = r
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(Array, self).validate(obj, value)

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
        super(Array, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return np.copy(self.default_value)


class PandasType(SciType):

    """A pandas dataframe or series trait type."""

    info_text = "a pandas dataframe or series"

    klass = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super(PandasType, self).validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(PandasType, self).validate(obj, value)

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
            raise TraitError("The klass attribute must be a class" " not: %r" % klass)
        if default_value is Empty:
            default_value = klass()
        elif default_value is not None and default_value is not Undefined:
            default_value = klass(default_value)
        super(PandasType, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return self.default_value.copy()


class DataFrame(PandasType):

    """A pandas dataframe trait type."""

    info_text = "a pandas dataframe"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import pandas as pd

            kwargs["klass"] = pd.DataFrame
        super(DataFrame, self).__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


class Series(PandasType):

    """A pandas series trait type."""

    info_text = "a pandas series"
    dtype = None

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import pandas as pd

            kwargs["klass"] = pd.Series
        super(Series, self).__init__(
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
            return super(XarrayType, self).validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(XarrayType, self).validate(obj, value)

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
            raise TraitError("The klass attribute must be a class" " not: %r" % klass)
        if default_value is Empty:
            default_value = klass()
        elif default_value is not None and default_value is not Undefined:
            default_value = klass(default_value)
        super(XarrayType, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return self.default_value.copy()


class Dataset(XarrayType):

    """An xarray dataset trait type."""

    info_text = "an xarray dataset"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import xarray as xr

            kwargs["klass"] = xr.Dataset
        super(Dataset, self).__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


class DataArray(XarrayType):

    """An xarray dataarray trait type."""

    info_text = "an xarray dataarray"
    dtype = None

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            import xarray as xr

            kwargs["klass"] = xr.DataArray
        super(DataArray, self).__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )
        self.dtype = dtype
