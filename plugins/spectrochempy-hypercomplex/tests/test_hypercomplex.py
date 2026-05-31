# ruff: noqa: S101
import numpy as np
import pytest


class TestHyperAccessor:
    def test_accessor_registered(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0])
        assert hasattr(ds, "hyper")

    def test_is_quaternion_false_for_real(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0])
        assert ds.hyper.is_quaternion is False

    def test_set_quaternion_creates_quaternion_dtype(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0, 4.0])
        result = ds.hyper.set_quaternion()
        assert result._data.dtype.name == "quaternion"

    def test_component_real_data_noop(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0])
        result = ds.hyper.component("RR")
        assert result is not None

    def test_RR_RI_IR_II_raise_for_real(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0])
        for prop in ["RR", "RI", "IR", "II"]:
            with pytest.raises(TypeError):
                getattr(ds.hyper, prop)


class TestNDMathHandlers:
    def test_handler_registration(self):
        from spectrochempy.plugins import manager

        registry = manager.plugin_manager.registry
        assert registry.get_handler("ndmath.execution_branch") is not None
        assert registry.get_handler("ndmath.execute") is not None


class TestQuaternionFunctions:
    def test_as_quaternion_roundtrip(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion
        from spectrochempy_hypercomplex._quaternion import quat_as_complex_array

        w = np.array([1.0, 2.0, 3.0])
        x = np.array([0.5, 0.6, 0.7])
        q = as_quaternion(w, x, w, x)
        c1, c2 = quat_as_complex_array(q)
        assert np.allclose(c1, w + 1j * x)
        assert np.allclose(c2, w + 1j * x)

    def test_as_quaternion_2args(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion

        r = np.array([1.0, 2.0]) + 1j * np.array([0.5, 0.6])
        i = np.array([3.0, 4.0]) + 1j * np.array([0.7, 0.8])
        q = as_quaternion(r, i)
        assert q.dtype.name == "quaternion"
        assert q.shape == (2,)

    def test_as_quaternion_2d_shape_preserved(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion

        w = np.ones((3, 4))
        x = np.zeros((3, 4))
        q = as_quaternion(w, x, w, x)
        assert q.shape == (3, 4)

    def test_quat_as_complex_array_nd(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion
        from spectrochempy_hypercomplex._quaternion import quat_as_complex_array

        w = np.arange(24.0).reshape(2, 3, 4)
        x = np.arange(24.0, 48.0).reshape(2, 3, 4)
        q = as_quaternion(w, x, w, x)
        c1, c2 = quat_as_complex_array(q)
        assert c1.shape == (2, 3, 4)
        assert np.allclose(c1, w + 1j * x)

    def test_get_component_quaternion_2d(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion
        from spectrochempy_hypercomplex._quaternion import get_component

        w = np.ones((3, 4))
        x = np.full((3, 4), 2.0)
        y = np.full((3, 4), 3.0)
        z = np.full((3, 4), 4.0)
        q = as_quaternion(w, x, y, z)
        assert np.allclose(get_component(q, "RR"), w)
        assert np.allclose(get_component(q, "RI"), x)
        assert np.allclose(get_component(q, "IR"), y)
        assert np.allclose(get_component(q, "II"), z)

    def test_get_component_raises_bad_select(self):
        from spectrochempy_hypercomplex._quaternion import as_quaternion
        from spectrochempy_hypercomplex._quaternion import get_component

        q = as_quaternion(np.ones(3), np.zeros(3), np.ones(3), np.zeros(3))
        with pytest.raises(ValueError, match="cannot interpret"):
            get_component(q, "RRI")

    def test_data_remains_real_noop(self):
        from spectrochempy_hypercomplex._quaternion import get_component

        arr = np.array([1.0, 2.0, 3.0])
        result = get_component(arr, "RR")
        assert np.array_equal(result, arr)

    def test_interleaved2complex(self):
        from spectrochempy_hypercomplex._quaternion import interleaved2complex

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = interleaved2complex(data)
        expected = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        assert np.allclose(result, expected)
