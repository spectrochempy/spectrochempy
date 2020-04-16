# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (c) IPython Development Team.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of traittypes nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ======================================================================================================================
"""
# Scipy Trait Types

Trait types for NumPy, SciPy and friends

## Goals

Provide a reference implementation of trait types for common data structures
used in the scipy stack such as
 - [numpy](https://github.com/numpy/numpy) arrays
 - [pandas](https://github.com/pydata/pandas) and [xarray](https://github.com/pydata/xarray) data structures

which are out of the scope of the main [traitlets](https://github.com/ipython/traitlets)
project but are a common requirement to build applications with traitlets in
combination with the scipy stack.

Another goal is to create adequate serialization and deserialization routines
for these trait types to be used with the [ipywidgets](https://github.com/ipython/ipywidgets)
project (`to_json` and `from_json`). These could also return a list of binary
buffers as allowed by the current messaging protocol.

## Installation


Using `pip` :

Make sure you have [pip installed](https://pip.readthedocs.org/en/stable/installing/) and run :

```
pip install traittypes
```

Using `conda` :

```
conda install -c conda-forge traittypes
```

## Usage

`traittypes` extends the `traitlets` library with an implementation of trait types for numpy arrays, pandas dataframes and pandas series.
 - `traittypes` works around some limitations with numpy array comparison to only trigger change events when necessary.
 - `traittypes` also extends the traitlets API for adding custom validators to constained proposed values for the attribute.

For a general introduction to `traitlets`, check out the [traitlets documentation](https://traitlets.readthedocs.io/en/stable/).

### Example usage with a custom validator

```python
from traitlets import HasTraits, TraitError
from traittypes import Array

def shape(*dimensions):
    def validator(trait, value):
        if value.shape != dimensions :
            raise TraitError('Expected an of shape %s and got and array with shape %s' % (dimensions, value.shape))
        else :
            return value
    return validator

class Foo(HasTraits):
    bar = Array(np.identity(2)).valid(shape(2, 2))
foo = Foo()

foo.bar = [1, 2]  # Should raise a TraitError
```

"""

from traitlets import TraitType, TraitError, Undefined


class _DelayedImportError(object):
    def __init__(self, package_name):
        self.package_name = package_name

    def __getattribute__(self, name):
        package_name = super(_DelayedImportError, self).__getattribute__('package_name')
        raise RuntimeError('Missing dependency : %s' % package_name)


try:
    import numpy as np
except ImportError:
    np = _DelayedImportError('numpy')
try:
    import pandas as pd
except ImportError:
    pd = _DelayedImportError('pandas')


class SciType(TraitType):
    """A base trait type for numpy arrays, pandas dataframes and series."""

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
                    if value.shape != dimensions :
                        raise TraitError('Expected an of shape %s and got and array with shape %s' % (dimensions, value.shape))
                    else :
                        return value
                return validator

            class Foo(HasTraits):
                bar = Array(np.identity(2)).valid(shape(2, 2))
            foo = Foo()

            foo.bar = [1, 2]  # Should raise a TraitError
        """
        self.validators.extend(validators)
        return self


class Array(SciType):
    """A numpy array trait type."""

    info_text = 'a numpy array'
    dtype = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        try:
            if hasattr(value, 'dtype'):
                dtype = value.dtype
            else:
                dtype = None
            value = np.asarray(value, dtype=dtype)
            for validator in self.validators:
                value = validator(self, value)
            return value
        except (ValueError, TypeError) as e:
            raise TraitError(e)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if not np.array_equal(old_value, new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Undefined, allow_none=False, dtype=None, **kwargs):
        self.dtype = dtype
        if default_value is Undefined:
            default_value = np.array(0, dtype=self.dtype)
        elif default_value is not None:
            default_value = np.asarray(default_value, dtype=self.dtype)
        self.validators = []
        super(Array, self).__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None:
            return self.default_value
        else:
            return np.copy(self.default_value)


class DataFrame(SciType):
    """A pandas dataframe trait type."""

    info_text = 'a pandas dataframe'

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        try:
            value = pd.DataFrame(value)
            for validator in self.validators:
                value = validator(self, value)
            return value
        except (ValueError, TypeError) as e:
            raise TraitError(e)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (old_value is None and new_value is not None) or not old_value.equals(new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Undefined, allow_none=False, dtype=None, **kwargs):
        import pandas as pd
        self.dtype = dtype
        if default_value is Undefined:
            default_value = pd.DataFrame()
        elif default_value is not None:
            default_value = pd.DataFrame(default_value)
        self.validators = []
        super(DataFrame, self).__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None:
            return self.default_value
        else:
            return self.default_value.copy()


class Series(SciType):
    """A pandas series trait type."""

    info_text = 'a pandas series'

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        try:
            value = pd.Series(value)
            for validator in self.validators:
                value = validator(self, value)
            return value
        except (ValueError, TypeError) as e:
            raise TraitError(e)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (old_value is None and new_value is not None) or not old_value.equals(new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Undefined, allow_none=False, dtype=None, **kwargs):
        import pandas as pd
        self.dtype = dtype
        if default_value is Undefined:
            default_value = pd.Series()
        elif default_value is not None:
            default_value = pd.Series(default_value)
        self.validators = []
        super(Series, self).__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None:
            return self.default_value
        else:
            return self.default_value.copy()
