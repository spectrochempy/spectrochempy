# ruff: noqa: S101
import re
from importlib.metadata import version

import numpy as np
import pytest
from spectrochempy_hypercomplex import HyperComplexPlugin

import spectrochempy as scp
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness
from spectrochempy.utils.print import _format_array_values
from spectrochempy.utils.print import pstr


class TestPluginLifecycle:
    def test_import(self):
        import spectrochempy_hypercomplex  # noqa: F401

    def test_plugin_metadata(self):
        plugin = HyperComplexPlugin()
        assert plugin.name == "hypercomplex"
        assert plugin.description
        assert PluginCapability.ACCESSOR in plugin.capabilities

    def test_plugin_version(self):
        plugin = HyperComplexPlugin()
        assert plugin.version
        assert isinstance(plugin.version, str)
        pkg_ver = version("spectrochempy-hypercomplex")
        assert plugin.version == pkg_ver, (
            f"Plugin version {plugin.version} != package version {pkg_ver}. "
            "Run `pip install -e` to sync."
        )

    def test_plugin_compatibility(self):
        plugin = HyperComplexPlugin()
        issues = check_plugin_compatibility(plugin)
        assert not issues, f"Compatibility issues: {issues}"

    def test_registration(self):
        harness = PluginTestHarness()
        harness.register(HyperComplexPlugin())

        accessors = list(harness.registry.available_accessors)
        assert "hyper" in accessors

    def test_lifecycle_state(self):
        harness = PluginTestHarness()
        harness.register(HyperComplexPlugin())
        assert harness.get_plugin_state("hypercomplex") == PluginState.ACTIVE

    def test_accessor_registered_via_plugin(self):
        harness = PluginTestHarness()
        harness.register(HyperComplexPlugin())

        acc = harness.get_accessor("hyper")
        assert acc is not None
        assert acc["obj"] is not None
        assert acc["metadata"]["plugin"] == "hypercomplex"
        assert acc["metadata"]["namespace"] == "hypercomplex"


class TestHyperAccessor:
    def test_accessor_registered(self):
        ds = scp.NDDataset([1.0, 2.0, 3.0])
        assert hasattr(ds, "hyper")

    def test_is_quaternion_false_for_real(self):
        ds = scp.NDDataset([1.0, 2.0, 3.0])
        assert ds.hyper.is_quaternion is False

    def test_set_quaternion_creates_quaternion_dtype(self):
        ds = scp.NDDataset([1.0, 2.0, 3.0, 4.0])
        result = ds.hyper.set_quaternion()
        assert result._data.dtype.name == "quaternion"

    def test_component_real_data_noop(self):
        ds = scp.NDDataset([1.0, 2.0, 3.0])
        result = ds.hyper.component("RR")
        assert result is not None

    def test_RR_RI_IR_II_raise_for_real(self):
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
        assert registry.get_handler("display.array_values") is not None
        assert registry.get_handler("display.complex_dim_flags") is not None


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


def _make_hypercomplex_dataset():
    ds = scp.NDDataset(np.arange(16.0).reshape(4, 4))
    return ds.hyper.set_quaternion()


class TestHypercomplexDisplayRegression:
    def test_terminal_display_restores_component_labels(self):
        ds = _make_hypercomplex_dataset()

        rendered = ds._str_value()

        for label in ["RR", "RI", "IR", "II"]:
            assert label in rendered
        assert "quaternion(" not in rendered

    def test_detailed_display_uses_component_blocks(self):
        ds = _make_hypercomplex_dataset()

        rendered = pstr(ds)

        for label in ["RR", "RI", "IR", "II"]:
            assert label in rendered
        assert "quaternion(" not in rendered

    def test_html_display_restores_component_labels(self):
        ds = _make_hypercomplex_dataset()

        html = ds._repr_html_()

        for label in ["RR", "RI", "IR", "II"]:
            assert label in html
        assert "quaternion(" not in html

    def test_component_values_match_accessor_components_in_terminal_and_html(self):
        ds = _make_hypercomplex_dataset()

        rendered = ds._str_value()
        html = ds._repr_html_()
        normalize_text = lambda s: re.sub(r"\n\s+", "\n", s)
        normalize_html = lambda s: re.sub(r"<br/>\s+", "<br/>", s)

        for label in ["RR", "RI", "IR", "II"]:
            component = np.asarray(getattr(ds.hyper, label))
            expected = _format_array_values(
                component,
                dtype=component.dtype,
                sep="\n",
                prefix=label,
                units="",
            )
            assert normalize_text(expected) in normalize_text(rendered)
            assert normalize_html(expected.replace("\n", "<br/>")) in normalize_html(
                html
            )

    def test_shape_annotation_marks_quaternion_dimensions_as_complex(self):
        ds = _make_hypercomplex_dataset()

        shape = ds._str_shape()

        assert "(y:4(complex), x:2(complex))" in shape
