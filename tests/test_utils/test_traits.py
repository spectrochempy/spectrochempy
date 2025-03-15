# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import unittest

import numpy as np
import traitlets as tr

from spectrochempy.utils.traits import CoordType
from spectrochempy.utils.traits import NDDatasetType
from spectrochempy.utils.traits import PositiveInteger
from spectrochempy.utils.traits import PositiveOddInteger
from spectrochempy.utils.traits import SpectroChemPyType


class DummyClass:
    """A dummy class for testing traits."""

    def __init__(self, *args, **kwargs):
        pass


class _A_Model(tr.HasTraits):
    """A model to test traits validation."""

    spectro_trait = tr.Instance(klass=object, allow_none=True)
    nddata_trait = tr.Instance(klass=object, allow_none=True)
    coord_trait = tr.Instance(klass=object, allow_none=True)
    pos_int = PositiveInteger()
    pos_odd_int = PositiveOddInteger()


class TestSpectroChemPyType(unittest.TestCase):
    def test_init(self):
        """Test initialization of SpectroChemPyType."""
        # Test valid initialization with klass
        trait = SpectroChemPyType(klass=DummyClass)
        self.assertEqual(trait.klass, DummyClass)

        # Test initialization with default value
        obj = DummyClass()
        trait = SpectroChemPyType(default_value=obj, klass=DummyClass)
        self.assertIsInstance(trait.default_value, DummyClass)

        # Test error when klass is not provided
        with self.assertRaises(tr.TraitError):
            SpectroChemPyType()

        # Test error when klass is not a class
        with self.assertRaises(tr.TraitError):
            SpectroChemPyType(klass="not a class")

    def test_validate(self):
        """Test validation in SpectroChemPyType."""
        model = _A_Model()
        trait = SpectroChemPyType(klass=DummyClass)

        # Test validation of None
        with self.assertRaises(tr.TraitError):
            trait.validate(model, None)

        # Test validation with allow_none=True
        trait_allow_none = SpectroChemPyType(klass=DummyClass, allow_none=True)
        self.assertIsNone(trait_allow_none.validate(model, None))

        # Test validation of valid object
        obj = DummyClass()
        validated = trait.validate(model, obj)
        self.assertIsInstance(validated, DummyClass)

        # Test conversion via klass constructor
        class ConvertibleClass:
            def __init__(self, value=None):
                pass

        trait_convert = SpectroChemPyType(klass=ConvertibleClass)
        validated = trait_convert.validate(model, "test")
        self.assertIsInstance(validated, ConvertibleClass)

    def test_make_dynamic_default(self):
        """Test make_dynamic_default in SpectroChemPyType."""

        class CopyableClass:
            def __init__(self, value=None):
                pass

            def copy(self):
                return CopyableClass()

        obj = CopyableClass()
        trait = SpectroChemPyType(default_value=obj, klass=CopyableClass)

        # Test making dynamic default
        default = trait.make_dynamic_default()
        self.assertIsInstance(default, CopyableClass)
        self.assertIsNot(default, obj)  # Should be a copy

        # Test None and Undefined handling
        trait_none = SpectroChemPyType(
            default_value=None, klass=CopyableClass, allow_none=True
        )
        self.assertIsNone(trait_none.make_dynamic_default())


class TestNDDatasetType(unittest.TestCase):
    def setUp(self):
        """Import NDDataset here to avoid import issues during test collection."""
        from spectrochempy.core.dataset.nddataset import NDDataset

        self.NDDataset = NDDataset

    def test_init(self):
        """Test initialization of NDDatasetType."""
        trait = NDDatasetType()
        self.assertEqual(trait.klass, self.NDDataset)

        # Test with dtype
        trait_with_dtype = NDDatasetType(dtype=np.float64)
        self.assertEqual(trait_with_dtype.metadata["dtype"], np.float64)


class TestCoordType(unittest.TestCase):
    def setUp(self):
        """Import Coord here to avoid import issues during test collection."""
        from spectrochempy.core.dataset.coord import Coord

        self.Coord = Coord

    def test_init(self):
        """Test initialization of CoordType."""
        trait = CoordType()
        self.assertEqual(trait.klass, self.Coord)

        # Test with dtype
        trait_with_dtype = CoordType(dtype=np.float32)
        self.assertEqual(trait_with_dtype.metadata["dtype"], np.float32)


class TestPositiveInteger(unittest.TestCase):
    def test_validate(self):
        """Test validation in PositiveInteger."""
        model = _A_Model()

        trait = PositiveInteger()

        # Test validation of positive integers
        validated = trait.validate(model, 5)
        self.assertEqual(validated, 5)

        validated = trait.validate(model, 0)
        self.assertEqual(validated, 0)

        # Test validation error for negative integers
        with self.assertRaises(tr.TraitError):
            trait.validate(model, -1)

        # Test None handling
        with self.assertRaises(tr.TraitError):
            trait.validate(model, None)

        trait_allow_none = PositiveInteger(allow_none=True)
        self.assertIsNone(trait_allow_none.validate(model, None))


class TestPositiveOddInteger(unittest.TestCase):
    def test_validate(self):
        """Test validation in PositiveOddInteger."""
        model = _A_Model()
        trait = PositiveOddInteger()

        # Test validation of positive odd integers
        validated = trait.validate(model, 1)
        self.assertEqual(validated, 1)

        validated = trait.validate(model, 3)
        self.assertEqual(validated, 3)

        # Test validation error for negative integers
        with self.assertRaises(tr.TraitError):
            trait.validate(model, -1)

        # Test validation error for even integers
        with self.assertRaises(tr.TraitError):
            trait.validate(model, 2)

        # Test None handling
        with self.assertRaises(tr.TraitError):
            trait.validate(model, None)

        trait_allow_none = PositiveOddInteger(allow_none=True)
        self.assertIsNone(trait_allow_none.validate(model, None))


if __name__ == "__main__":
    unittest.main()
