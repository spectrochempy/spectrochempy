# ruff: noqa: S101
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
        # After conversion, the data should have a quaternion dtype
        assert result._data.dtype.name == "quaternion"

    def test_component_real_data_noop(self):
        import spectrochempy as scp

        ds = scp.NDDataset([1.0, 2.0, 3.0])
        # For real data, get_component returns the array unchanged
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
