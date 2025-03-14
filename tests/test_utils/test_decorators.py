# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import inspect
import warnings

import traitlets as tr

import spectrochempy as scp
from spectrochempy.utils.decorators import _wrap_ndarray_output_to_nddataset
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.decorators import signature_has_configurable_traits


class TestDeprecated:
    def test_deprecated_function(self):
        @deprecated(replace="new_function", removed="1.0")
        def old_function():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message)
            assert "new_function" in str(w[0].message)
            assert "1.0" in str(w[0].message)
            assert result == 42

    def test_deprecated_attribute(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated("old_attr", replace="new_attr", removed="1.0")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "`old_attr` attribute" in str(w[0].message)


class TestSignatureHasConfigurableTraits:
    def test_signature_has_configurable_traits(self):
        @signature_has_configurable_traits
        class MyClass(tr.HasTraits):
            """
            My test class.

            Parameters
            ----------
            name : str
                The name parameter
            """

            value = tr.Int(0, config=True, help="An integer value")
            name = tr.Unicode("default", config=True, help="A string name")

            def __init__(self, name=None, **kwargs):
                super().__init__(**kwargs)
                if name is not None:
                    self.name = name

        # Check that the signature includes the trait parameters
        sig = inspect.signature(MyClass)
        assert "name" in sig.parameters
        assert "value" in sig.parameters

        # Check that the docstring was updated
        assert "value : `int`" in MyClass.__doc__
        assert "name : `str`" in MyClass.__doc__

        # Check instance creation with trait parameters
        instance = MyClass(value=42, name="test")
        assert instance.value == 42
        assert instance.name == "test"


class TestWrapNdarrayOutput:
    def test_wrap_ndarray_output(self):
        # Create a simple class with a method that returns numpy arrays
        class TestModel:
            def __init__(self):
                self.name = "test_model"
                self._X = scp.NDDataset([[1, 2, 3]], title="Test Dataset")

            def _restore_masked_data(self, x, **kwargs):
                return x

            @_wrap_ndarray_output_to_nddataset(meta_from="_X")
            def transform(self, X=None):
                if X is None:
                    X = self._X
                return X.data * 2

        model = TestModel()
        result = model.transform()

        # Check that the result is an NDDataset
        assert isinstance(result, scp.NDDataset)
        # Check that metadata was properly transferred
        assert result.title == "Test Dataset"
        # Check that the data was properly transformed
        assert list(result.data) == [2.0, 4.0, 6.0]
        # Check that the name was properly set
        assert "test_model.transform" in result.name
