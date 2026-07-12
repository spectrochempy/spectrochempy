# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Integration tests for the plugin system (lazy loading, discovery, user-facing errors)."""

from __future__ import annotations

import importlib.metadata as im
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import spectrochempy
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.api.plugins.validation import check_plugin_contributions
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.deps import MissingPluginNamespaceError
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry

_CANTERA_INSTALLED = importlib.util.find_spec("spectrochempy_cantera") is not None
_IRIS_INSTALLED = importlib.util.find_spec("spectrochempy_iris") is not None
_TENSOR_INSTALLED = importlib.util.find_spec("spectrochempy_tensor") is not None
_TENSORLY_INSTALLED = importlib.util.find_spec("tensorly") is not None

# ------------------------------------------------------------------
# Fake plugins for integration testing
# ------------------------------------------------------------------


class FakeReaderPlugin(SpectroChemPyPlugin):
    name = "fake_reader"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_readers(self) -> list[dict]:
        def read_fake(path: str) -> str:
            return f"fake data from {path}"

        return [
            {
                "name": "fake",
                "func": read_fake,
                "description": "Read fake format",
                "extensions": [".fake"],
            }
        ]


class FakeNamespacePlugin(SpectroChemPyPlugin):
    name = "fakenamespace"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_readers(self) -> list[dict]:
        def read_fake_ns(path: str) -> str:
            return f"fake ns data from {path}"

        return [
            {
                "name": "fakename",
                "func": read_fake_ns,
                "description": "Read fake ns format",
                "namespace": "fakenamespace",
                "extensions": [".fns"],
            }
        ]


class FakeAccessorPlugin(SpectroChemPyPlugin):
    name = "fakeaccessor"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_accessors(self) -> list[dict]:
        def my_accessor(dataset: Any, factor: float = 2.0) -> str:
            return f"accessor applied with factor {factor}"

        return [
            {
                "namespace": "fakeaccessor",
                "name": "transform",
                "func": my_accessor,
                "description": "Fake accessor transform",
            }
        ]


class InvalidPlugin:
    """Plugin without name/version/api_version — malformed."""

    def register(self, registry: Any) -> None:
        registry.register_reader("bad", lambda x: x)


# ------------------------------------------------------------------
# A. Import core
# ------------------------------------------------------------------


class TestCoreImport:
    def test_import_spectrochempy(self):
        """Import spectrochempy works without plugin dependencies."""

        assert spectrochempy.__version__ is not None

    def test_access_nddataset(self):
        """scp.NDDataset is accessible."""

        assert spectrochempy.NDDataset is not None

    def test_access_read(self):
        """scp.read is accessible (core IO function)."""

        assert callable(spectrochempy.read)

    def test_access_read_omnic(self):
        """scp.read_omnic is accessible (built-in reader)."""

        assert callable(spectrochempy.read_omnic)

    def test_import_spectrochempy_does_not_import_iris_plugin(self):
        """Importing the core package alone does not import the IRIS plugin."""

        code = (
            "import sys; import spectrochempy; "
            "raise SystemExit("
            "any(name.startswith('spectrochempy_iris') for name in sys.modules)"
            ")"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


# ------------------------------------------------------------------
# G. Lazy loading non-regression
# ------------------------------------------------------------------


class TestLazyLoadingNonRegression:
    """
    Verify that certain operations do NOT load heavy plugin modules.

    Each test runs in a fresh subprocess and snapshots ``sys.modules`` **after**
    the setup phase so that only the targeted operation is checked.
    """

    LAZY_MODULES = [
        "spectrochempy_nmr",
        "spectrochempy_nmr.readers.read_topspin",
        "spectrochempy_cantera",
        "spectrochempy_cantera._pfr",
        "spectrochempy_iris",
        "spectrochempy_iris._core",
        "spectrochempy_tensor",
        "spectrochempy_tensor.decompositions.cp",
        "cantera",
    ]

    @staticmethod
    def _bad_imports_filter():
        parts = [
            f"mod == {m!r} or mod.startswith({m!r} + '.')"
            for m in TestLazyLoadingNonRegression.LAZY_MODULES
        ]
        return " or ".join(parts)

    def _run(self, setup: str, operation: str, description: str) -> None:
        import subprocess
        import sys

        bad = self._bad_imports_filter()
        # We snapshot *after* setup so that any modules legitimately loaded
        # during setup (e.g. by plugin_manager.discover()) do not cause a
        # false positive.  Only the operation is checked for new additions.
        code = f"""
import sys
{setup}
after_setup = {{mod for mod in sys.modules if {bad}}}
{operation}
after = {{mod for mod in sys.modules if {bad}}}
new = after - after_setup
if new:
    print(f"EAGER ({description}): {{new}}")
    raise SystemExit(1)
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"{description}: {result.stderr or result.stdout}"

    def test_import_spectrochempy(self):
        """``import spectrochempy`` does not load plugin modules."""
        self._run(
            setup="",
            operation="import spectrochempy",
            description="import spectrochempy",
        )

    def test_discover(self):
        """``plugin_manager.discover()`` loads plugin modules (expected)."""
        # discover() calls ep.load() which imports the actual plugin
        # packages.  This is by design — it is the explicit API for
        # loading all registered plugin entry points.
        import subprocess
        import sys

        code = """
import importlib.metadata as im
import sys
import spectrochempy
spectrochempy.plugin_manager.discover()
expected_mods = set()
for ep in im.entry_points(group="spectrochempy.plugins"):
    mod_name = ep.value.split(":")[0]
    expected_mods.add(mod_name)
loaded = set(sys.modules)
missing = expected_mods - loaded
assert not missing, f"Missing modules: {missing}"
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_namespace_access(self):
        """Official/experimental namespace access does not load plugin modules."""
        self._run(
            setup="import spectrochempy",
            operation=(
                "_ = spectrochempy.iris\n"
                "_ = spectrochempy.nmr\n"
                "_ = spectrochempy.tensor\n"
                "_ = spectrochempy.cantera"
            ),
            description="namespace access",
        )

    def test_submodule_import(self):
        """``import spectrochempy.<ns>`` does not load plugin modules."""
        for ns in ("iris", "nmr", "tensor", "cantera"):
            self._run(
                setup="import spectrochempy",
                operation=f"import spectrochempy.{ns}",
                description=f"import spectrochempy.{ns}",
            )

    def test_repr_does_not_resolve(self):
        """``repr(proxy)`` does not load the underlying sub-module."""
        self._run(
            setup=(
                "import spectrochempy\n"
                "spectrochempy.plugin_manager.discover()\n"
                "proxy = spectrochempy.nmr.read_topspin"
            ),
            operation="_ = repr(proxy)",
            description="repr",
        )

    def test_from_import_resolves(self):
        """``from spectrochempy.nmr import read_topspin`` returns a callable."""
        import subprocess
        import sys

        code = """
import sys
import spectrochempy as scp
scp.plugin_manager.discover()
from spectrochempy.nmr import read_topspin
# The proxy should be callable but should NOT have loaded the
# read_topspin sub-module (that only happens on actual resolution).
assert callable(read_topspin), "read_topspin should be callable"
if "spectrochempy_nmr.readers.read_topspin" in sys.modules:
    print("read_topspin sub-module was eagerly resolved")
    raise SystemExit(1)
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


# ------------------------------------------------------------------
# H. Missing plugin — clear actionable error
# ------------------------------------------------------------------


class TestMissingPluginError:
    """Clear error messages when an official plugin is not installed."""

    def _make_missing_namespace(self, ns_name: str, monkeypatch):
        """Create a PluginNamespace with discovery disabled (simulating missing plugin)."""
        from spectrochempy.plugins.manager import PluginManager
        from spectrochempy.plugins.namespace import PluginNamespace

        monkeypatch.setattr(im, "entry_points", lambda group=None: [])
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        return PluginNamespace(ns_name, manager, registry)

    def _simulate_no_discovered_plugins(self, monkeypatch):
        """Point the package-level plugin manager at an empty registry."""
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        monkeypatch.setattr(im, "entry_points", lambda group=None: [])
        monkeypatch.setattr(spectrochempy, "registry", registry)
        monkeypatch.setattr(spectrochempy, "plugin_manager", manager)

    def test_missing_known_namespace_attribute_error(self, monkeypatch):
        """Accessing an attribute on a missing known namespace raises MissingPluginNamespaceError."""
        from spectrochempy.plugins.deps import MissingPluginNamespaceError

        for ns_name in ("nmr", "iris", "tensor", "carroucell", "cantera"):
            ns = self._make_missing_namespace(ns_name, monkeypatch)
            with pytest.raises(MissingPluginNamespaceError) as excinfo:
                _ = ns.some_attribute
            msg = str(excinfo.value)
            assert "pip install" in msg, f"{ns_name}: {msg}"

    def test_missing_unknown_namespace_generic_error(self, monkeypatch):
        """Unknown namespace without hint raises generic AttributeError."""
        ns = self._make_missing_namespace("nonexistent", monkeypatch)
        with pytest.raises(AttributeError) as excinfo:
            _ = ns.something
        msg = str(excinfo.value)
        assert "has no attribute" in msg

    def test_missing_official_root_symbol_guides_install(self, monkeypatch):
        """A missing official plugin root symbol gives an actionable AttributeError."""
        self._simulate_no_discovered_plugins(monkeypatch)

        with pytest.raises(AttributeError) as excinfo:
            _ = spectrochempy.IRIS

        msg = str(excinfo.value)
        assert "module 'spectrochempy' has no attribute 'IRIS'" in msg
        assert "Did you mean:" in msg
        assert "scp.iris.IRIS" in msg
        assert "The official IRIS plugin is not installed" in msg
        assert "pip install spectrochempy-iris" in msg
        assert "pip install spectrochempy[plugins]" in msg

    def test_missing_official_root_symbol_preserves_hasattr(self, monkeypatch):
        """The richer AttributeError remains compatible with hasattr()."""
        self._simulate_no_discovered_plugins(monkeypatch)

        assert hasattr(spectrochempy, "IRIS") is False
        assert hasattr(spectrochempy, "CP") is False

    def test_experimental_root_symbol_has_no_special_hint(self, monkeypatch):
        """Experimental plugin symbols are not promoted by install guidance."""
        from spectrochempy.plugins.features import plugin_symbol_install_hint

        assert plugin_symbol_install_hint("PFR") is None
        self._simulate_no_discovered_plugins(monkeypatch)

        with pytest.raises(AttributeError) as excinfo:
            _ = spectrochempy.CanteraReactor

        msg = str(excinfo.value)
        assert "module 'spectrochempy' has no attribute 'CanteraReactor'" in msg
        assert "Did you mean:" not in msg
        assert "spectrochempy-cantera" not in msg


# ------------------------------------------------------------------
# I. Deprecated root aliases
# ------------------------------------------------------------------


class TestDeprecatedRootAliases:
    """``scp.IRIS``, ``scp.PFR`` etc. emit a DeprecationWarning."""

    @pytest.mark.skipif(not _IRIS_INSTALLED, reason="iris plugin not installed")
    def test_iris_root_alias_deprecated(self):
        """scp.IRIS emits DeprecationWarning pointing to scp.iris.IRIS."""
        import warnings

        import spectrochempy as scp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = scp.IRIS
            assert obj is not None, "IRIS should still be accessible"
            assert any(
                "deprecated" in str(msg.message).lower() for msg in w
            ), f"No deprecation warning: {[str(m.message) for m in w]}"

    @pytest.mark.skipif(not _IRIS_INSTALLED, reason="iris plugin not installed")
    def test_iris_kernel_root_alias_deprecated(self):
        """scp.IrisKernel emits DeprecationWarning."""
        import warnings

        import spectrochempy as scp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = scp.IrisKernel
            assert obj is not None
            assert any("deprecated" in str(msg.message).lower() for msg in w)

    @pytest.mark.skipif(not _CANTERA_INSTALLED, reason="cantera plugin not installed")
    def test_pfr_root_alias_deprecated(self):
        """scp.PFR emits DeprecationWarning."""
        import warnings

        import spectrochempy as scp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = scp.PFR
            assert obj is not None
            assert any("deprecated" in str(msg.message).lower() for msg in w)

    @pytest.mark.skipif(not _IRIS_INSTALLED, reason="iris plugin not installed")
    def test_namespaced_access_no_warning(self):
        """scp.iris.IRIS does NOT emit DeprecationWarning."""
        import warnings

        import spectrochempy as scp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = scp.iris.IRIS
            assert obj is not None
            dep_msgs = [
                str(m.message) for m in w if "deprecated" in str(m.message).lower()
            ]
            assert len(dep_msgs) == 0, f"Unexpected warnings: {dep_msgs}"

    @pytest.mark.skipif(
        not (_TENSOR_INSTALLED and _TENSORLY_INSTALLED),
        reason="tensor plugin dependencies not installed",
    )
    def test_cp_root_alias_deprecated(self):
        """scp.CP emits DeprecationWarning pointing to scp.tensor.CP."""
        import warnings

        import spectrochempy as scp

        scp._EMITTED_PLUGIN_ROOT_WARNINGS.discard("CP")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = scp.CP
            assert obj is not None, "CP should still be accessible"
            assert any(
                "scp.tensor.CP" in str(msg.message) for msg in w
            ), f"No CP deprecation warning: {[str(m.message) for m in w]}"


class TestSubmoduleImport:
    """``from spectrochempy.<ns> import X`` via PluginNamespaceModule."""

    @pytest.mark.skipif(not _IRIS_INSTALLED, reason="iris plugin not installed")
    def test_lazy_loading_on_import_with_access(self):
        """Accessing IRIS after import spectrochempy.iris loads _core."""
        import subprocess
        import sys

        code = """
import sys
import spectrochempy.iris as iris_mod
_ = iris_mod.IRIS
expected = [\"spectrochempy_iris._core\"]
for mod in expected:
    if mod not in sys.modules:
        print(f\"Missing: {mod}\")
        raise SystemExit(1)
unexpected = [\"spectrochempy_cantera._pfr\", \"spectrochempy_nmr.readers.read_topspin\"]
for mod in unexpected:
    if mod in sys.modules:
        print(f\"Unexpected eager: {mod}\")
        raise SystemExit(1)
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    @pytest.mark.skipif(not _IRIS_INSTALLED, reason="iris plugin not installed")
    def test_submodule_dir(self):
        """dir() on the pseudo-module works and includes plugin names."""
        import spectrochempy.iris as iris_mod

        names = dir(iris_mod)
        assert "IRIS" in names
        assert "IrisKernel" in names

    def test_known_missing_submodule_import_succeeds(self):
        """``import spectrochempy.<ns>`` succeeds (returns PluginNamespaceModule)."""
        import subprocess
        import sys

        code = """
import importlib.metadata as im
import sys
import spectrochempy as scp
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry

# Replace manager/registry with empty ones
registry = PluginRegistry()
manager = PluginManager(registry=registry)
scp.plugin_manager = manager
scp.registry = registry
im.entry_points = lambda group=None: []

# Remove existing namespace modules from sys.modules and re-register
from spectrochempy.plugins.features import KNOWN_PLUGIN_NAMESPACES
for ns in KNOWN_PLUGIN_NAMESPACES:
    sys.modules.pop(f"spectrochempy.{ns}", None)
from spectrochempy.plugins.namespace import register_namespace_modules
register_namespace_modules()

# import should succeed (returns PluginNamespaceModule)
import spectrochempy.nmr as nmr_mod
assert nmr_mod is not None

# Attribute access on the namespace should raise
from spectrochempy.plugins.deps import MissingPluginNamespaceError
try:
    _ = nmr_mod.read_topspin
except MissingPluginNamespaceError:
    pass
else:
    raise AssertionError("attribute access should raise MissingPluginNamespaceError")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


# ------------------------------------------------------------------
# E. Missing plugin / stub
# ------------------------------------------------------------------


class TestMissingPlugin:
    def test_missing_reader_clear_error(self, monkeypatch):
        """Accessing scp.read_topspin without NMR plugin gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(im, "entry_points", lambda group=None: [])
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)
        monkeypatch.setitem(
            spectrochempy._LAZY_IMPORTS,
            "read_topspin",
            "spectrochempy.missing.lazy_import_entry",
        )

        with pytest.raises(MissingPluginError) as excinfo:
            spectrochempy.read_topspin("missing")
        message = str(excinfo.value)
        assert "spectrochempy-nmr" in message
        assert "spectrochempy[nmr]" in message

    def test_unknown_plugin_reader_is_not_stubbed(self, monkeypatch):
        """Unknown read_* attributes fail normally instead of creating stubs."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)
        monkeypatch.delitem(
            spectrochempy.__dict__,
            "read_totally_unknown_format",
            raising=False,
        )

        with pytest.raises(AttributeError, match="has no attribute") as excinfo:
            _ = spectrochempy.read_totally_unknown_format

        assert not isinstance(excinfo.value, MissingPluginError)
        assert "read_totally_unknown_format" not in spectrochempy.__dict__

    def test_missing_namespace_clear_error(self, monkeypatch):
        """Accessing a sub-attribute on a missing namespace gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(im, "entry_points", lambda group=None: [])
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)
        monkeypatch.setattr(spectrochempy.plugins.manager, "plugin_manager", pm)

        # Remove existing sys.modules entries so they get recreated with
        # the monkeypatched manager.
        for ns in ("nmr", "iris", "tensor", "cantera"):
            key = f"spectrochempy.{ns}"
            if key in sys.modules:
                del sys.modules[key]

        with pytest.raises(
            MissingPluginNamespaceError, match="requires the optional plugin"
        ):
            _ = spectrochempy.nmr.read_topspin

    def test_missing_iris_namespace_clear_error(self, monkeypatch):
        """Attribute access on a missing iris namespace gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(im, "entry_points", lambda group=None: [])
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)
        monkeypatch.setattr(spectrochempy.plugins.manager, "plugin_manager", pm)

        # Remove existing sys.modules entries so they get recreated with
        # the monkeypatched manager.
        for ns in ("nmr", "iris", "tensor", "cantera"):
            key = f"spectrochempy.{ns}"
            if key in sys.modules:
                del sys.modules[key]

        with pytest.raises(
            MissingPluginNamespaceError, match="spectrochempy-iris"
        ) as excinfo:
            _ = spectrochempy.iris.IRIS

        assert "spectrochempy[iris]" in str(excinfo.value)

    def test_missing_iris_from_import_clear_error(self):
        """``from spectrochempy import iris`` succeeds; attribute access gives the hint."""
        code = """
import importlib.metadata as im
import sys
import spectrochempy as scp
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry

# Replace manager/registry with empty ones
registry = PluginRegistry()
manager = PluginManager(registry=registry)
scp.plugin_manager = manager
scp.registry = registry
scp.plugins.manager.plugin_manager = manager
im.entry_points = lambda group=None: []

# Remove existing sys.modules entries so they get recreated with
# the monkeypatched manager.
from spectrochempy.plugins.features import KNOWN_PLUGIN_NAMESPACES
for ns in KNOWN_PLUGIN_NAMESPACES:
    sys.modules.pop(f"spectrochempy.{ns}", None)
from spectrochempy.plugins.namespace import register_namespace_modules
register_namespace_modules()

# from-import should succeed
from spectrochempy import iris
assert iris is not None

# Attribute access should raise the missing-plugin hint
from spectrochempy.plugins.deps import MissingPluginNamespaceError
try:
    _ = iris.IRIS
except MissingPluginNamespaceError as exc:
    message = str(exc)
    assert "spectrochempy-iris" in message
    assert "spectrochempy[iris]" in message
else:
    raise AssertionError("attribute access on missing iris namespace should raise")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_unknown_attribute_standard_error(self):
        """Accessing a truly unknown attribute gives a standard error."""

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = spectrochempy.nonexistent_attribute_xyz123


# ------------------------------------------------------------------
# F. Invalid plugin
# ------------------------------------------------------------------


class FailingRegisterPlugin:
    """Plugin whose imperative register() raises."""

    name = "fail-reg"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register(self, registry):
        msg = "registration failure"
        raise RuntimeError(msg)


class TestInvalidPlugin:
    def test_invalid_plugin_does_not_crash(self):
        """A malformed plugin does not crash the manager."""
        pm = PluginManager()
        plugin = InvalidPlugin()
        pm.register(plugin)

    def test_invalid_plugin_fails_gracefully(self):
        """A malformed plugin is tracked as FAILED."""
        pm = PluginManager()
        plugin = InvalidPlugin()
        pm.register(plugin)
        state = pm.get_plugin_state("invalidplugin")
        assert state is not None
        assert state.value == "failed"

    def test_valid_plugin_works_after_invalid(self):
        """A valid plugin can still register after an invalid one."""
        pm = PluginManager()
        pm.register(InvalidPlugin())
        pm.register(FakeReaderPlugin())
        assert pm.get_plugin_state("fake_reader").value == "active"

    def test_discover_does_not_crash_with_bad_entry_points(self, monkeypatch):
        """An entry point that loads a broken class does not crash discover()."""

        class BrokenEntryPoint:
            name = "broken_ep"
            value = "broken_module:BrokenPlugin"

            @staticmethod
            def load():
                msg = "broken module"
                raise ImportError(msg)

        original = im.entry_points

        def mock_entry_points(group=None):
            if group == ENTRY_POINT_GROUP:
                return [BrokenEntryPoint()]
            return original(group=group)

        monkeypatch.setattr(im, "entry_points", mock_entry_points)
        pm = PluginManager()
        pm.discover()  # must not raise

    def test_failed_plugin_not_available_via_get_plugin(self):
        """A FAILED plugin must not appear in get_plugin()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert pm.get_plugin("fail-reg") is None

    def test_failed_plugin_not_available_via_has_plugin(self):
        """A FAILED plugin must not appear in has_plugin()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert pm.has_plugin("fail-reg") is False

    def test_failed_plugin_not_in_list_plugins(self):
        """A FAILED plugin must not appear in list_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert "fail-reg" not in [
            p.name for p in pm.list_plugins() if hasattr(p, "name")
        ]

    def test_failed_plugin_not_active(self):
        """A FAILED plugin must not appear in get_active_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert "fail-reg" not in pm.get_active_plugins()

    def test_failed_plugin_reported_in_get_failed(self):
        """A FAILED plugin is still tracked via get_failed_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        failed = pm.get_failed_plugins()
        assert "fail-reg" in failed

    def test_load_plugin_returns_none_for_failed(self):
        """load_plugin returns None when the plugin fails to register."""
        pm = PluginManager()

        class AlwaysFails:
            name = "always-fails"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register(self, registry):
                msg = "always fails"
                raise RuntimeError(msg)

        result = pm.load_plugin("always-fails")
        assert result is None


# ------------------------------------------------------------------
# G. Dataset accessors
# ------------------------------------------------------------------


class TestDatasetAccessors:
    def test_accessor_not_duplicated_on_repeated_register(self):
        """Calling register() twice with the same plugin does not add duplicate accessors."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeAccessorPlugin()
        pm.register(plugin)
        assert len(registry.available_accessors) == 1
        pm.register(plugin)
        assert len(registry.available_accessors) == 1

    def test_accessor_does_not_become_reader(self):
        """An accessor is not accidentally exposed as a reader."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeAccessorPlugin()
        pm.register(plugin)
        assert registry.get_reader("transform") is None
        assert registry.get_accessor("fakeaccessor.transform") is not None


# ------------------------------------------------------------------
# H. Discovery state machine
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# I. __dir__ without side effects
# ------------------------------------------------------------------


class TestDirNoSideEffect:
    def test_dir_does_not_trigger_discovery(self, monkeypatch):
        """dir(scp) must not trigger plugin_manager.discover()."""

        discover_called = False
        original_discover = spectrochempy.plugin_manager.discover

        def tracking_discover():
            nonlocal discover_called
            discover_called = True
            original_discover()

        monkeypatch.setattr(spectrochempy.plugin_manager, "discover", tracking_discover)
        dir(spectrochempy)
        assert not discover_called, "dir(scp) triggered plugin_manager.discover()"


# ------------------------------------------------------------------
# J. Warnings for invalid contributions
# ------------------------------------------------------------------


class TestInvalidContributionWarnings:
    def test_warning_on_non_list_return(self, caplog):
        """register_readers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadReturnPlugin:
            name = "bad-return"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return "not a list"

        pm = PluginManager()
        pm.register(BadReturnPlugin())
        assert any(
            "bad-return" in msg and "register_readers" in msg for msg in caplog.messages
        )

    def test_warning_on_non_dict_item(self, caplog):
        """register_readers returning a non-dict item logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadItemPlugin:
            name = "bad-item"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return ["not a dict"]

        pm = PluginManager()
        pm.register(BadItemPlugin())
        assert any("bad-item" in msg and "is str" in msg for msg in caplog.messages)

    def test_warning_on_missing_keys(self, caplog):
        """register_readers with missing 'name'/'func' keys logs a warning."""

        caplog.set_level(logging.WARNING)

        class MissingKeysPlugin:
            name = "missing-keys"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return [{"description": "no name or func"}]

        pm = PluginManager()
        pm.register(MissingKeysPlugin())
        assert any(
            "missing-keys" in msg and "missing required keys" in msg
            for msg in caplog.messages
        )

    def test_warning_on_writers_non_list(self, caplog):
        """register_writers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadWritersPlugin:
            name = "bad-writers"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_writers(self):
                return 42

        pm = PluginManager()
        pm.register(BadWritersPlugin())
        assert any(
            "bad-writers" in msg and "register_writers" in msg
            for msg in caplog.messages
        )

    def test_warning_on_analyses_non_list(self, caplog):
        """register_analyses returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadAnalysesPlugin:
            name = "bad-analyses"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_analyses(self):
                return None

        pm = PluginManager()
        pm.register(BadAnalysesPlugin())
        assert any(
            "bad-analyses" in msg and "register_analyses" in msg
            for msg in caplog.messages
        )

    def test_warning_on_simulations_non_list(self, caplog):
        """register_simulations returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadSimPlugin:
            name = "bad-sim"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_simulations(self):
                return "invalid"

        pm = PluginManager()
        pm.register(BadSimPlugin())
        assert any(
            "bad-sim" in msg and "register_simulations" in msg
            for msg in caplog.messages
        )

    def test_warning_on_accessors_non_list(self, caplog):
        """register_accessors returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadAccessorPlugin:
            name = "bad-accessor"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_accessors(self):
                return [42]

        pm = PluginManager()
        pm.register(BadAccessorPlugin())
        assert any(
            "bad-accessor" in msg and "register_accessors" in msg
            for msg in caplog.messages
        )

    def test_warning_on_processors_non_list(self, caplog):
        """register_processors returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadProcPlugin:
            name = "bad-proc"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_processors(self):
                return (1, 2)

        pm = PluginManager()
        pm.register(BadProcPlugin())
        assert any(
            "bad-proc" in msg and "register_processors" in msg
            for msg in caplog.messages
        )

    def test_warning_on_visualizers_non_list(self, caplog):
        """register_visualizers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadVizPlugin:
            name = "bad-viz"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_visualizers(self):
                return [{"name": "foo"}]  # missing func

        pm = PluginManager()
        pm.register(BadVizPlugin())
        assert any(
            "bad-viz" in msg and "missing required keys" in msg
            for msg in caplog.messages
        )


class TestDiscoveryStateMachine:
    def test_discovery_state_transition(self):
        """Discovery goes NOT_DISCOVERED -> DISCOVERING -> DISCOVERED."""
        pm = PluginManager()
        assert pm._discovery_state == "not_discovered"
        pm.discover()
        assert pm._discovery_state == "discovered"

    def test_reentrant_discovery_safe(self):
        """Calling discover() during discover() is safe (no infinite loop)."""
        pm = PluginManager()

        class ReentrantPlugin:
            name = "reentrant"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register(self, registry):
                pm.discover()

        pm.register(ReentrantPlugin())
        assert pm._discovery_state == "discovered"

    def test_discovery_skipped_when_already_discovering(self):
        """Calling discover() while already discovering returns immediately."""
        pm = PluginManager()
        pm._discovering = True
        pm.discover()
        assert pm._discovery_state == "not_discovered"

    def test_registration_does_not_trigger_discovery(self):
        """Registering a plugin directly does not trigger entry point discovery."""

        original = im.entry_points
        call_count = 0

        def counting_mock(group=None):
            nonlocal call_count
            call_count += 1
            return original(group=group) if group != ENTRY_POINT_GROUP else []

        pm = PluginManager()
        pm.register(FakeReaderPlugin())
        assert call_count == 0


# ------------------------------------------------------------------
# K. Orphan contributions when register() fails
# ------------------------------------------------------------------


class TestOrphanContributions:
    """
    A plugin whose imperative register() fails must not leave
    declarative contributions (readers, writers, …) in the registry.
    """

    def test_reader_not_registered_when_register_fails(self):
        """register_readers() works but register() raises → no reader in registry."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class DeclareThenFailPlugin:
            name = "declare-then-fail"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return [{"name": "orphan-reader", "func": lambda p: p}]

            def register(self, registry):
                msg = "intentional failure"
                raise RuntimeError(msg)

        pm.register(DeclareThenFailPlugin())
        assert pm.get_plugin_state("declare-then-fail").value == "failed"
        # The reader must NOT be available
        assert registry.get_reader("orphan-reader") is None

    def test_writer_not_registered_when_register_fails(self):
        """register_writers() works but register() raises → no writer in registry."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class DeclareWriterThenFailPlugin:
            name = "writer-fail"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_writers(self):
                return [{"name": "orphan-writer", "func": lambda p: p}]

            def register(self, registry):
                msg = "intentional failure"
                raise RuntimeError(msg)

        pm.register(DeclareWriterThenFailPlugin())
        assert pm.get_plugin_state("writer-fail").value == "failed"
        assert registry.get_writer("orphan-writer") is None

    def test_accessor_not_registered_when_register_fails(self):
        """register_accessors() works but register() raises → no accessor in registry."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class DeclareAccessorThenFailPlugin:
            name = "accessor-fail"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_accessors(self):
                return [{"name": "orphan-accessor", "func": lambda p: p}]

            def register(self, registry):
                msg = "intentional failure"
                raise RuntimeError(msg)

        pm.register(DeclareAccessorThenFailPlugin())
        assert pm.get_plugin_state("accessor-fail").value == "failed"
        assert registry.get_accessor("accessor-fail.orphan-accessor") is None


# ------------------------------------------------------------------
# L. check_plugin_contributions does not double-execute hooks
# ------------------------------------------------------------------


class TestCheckPluginContributionsNoSideEffect:
    """
    check_plugin_contributions() validates hook presence only and must not
    execute declarative hooks.
    """

    def test_check_plugin_contributions_does_not_call_hooks(self):
        """Validation must not execute declarative hook methods."""

        class CountingPlugin:
            call_count = 0
            name = "counter-validation"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                self.call_count += 1
                return [{"name": "cnt", "func": lambda p: p}]

        plugin = CountingPlugin()
        issues = check_plugin_contributions(plugin)

        assert issues == []
        assert plugin.call_count == 0

    def test_register_calls_declarative_hooks_once(self):
        """Registering a plugin calls each declarative hook exactly once."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class CountingPlugin:
            call_count = 0
            name = "counter"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                self.call_count += 1
                return [{"name": "cnt", "func": lambda p: p}]

        plugin = CountingPlugin()
        pm.register(plugin)
        assert plugin.call_count == 1, "register_readers should be called exactly once"
        assert registry.get_reader("cnt") is not None


# ------------------------------------------------------------------
# M. load_plugin behaviour
# ------------------------------------------------------------------


class TestLoadPlugin:
    def test_load_plugin_returns_none_for_missing(self):
        """load_plugin('nonexistent') returns None."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        result = pm.load_plugin("nonexistent-plugin-name")
        assert result is None

    def test_load_plugin_failed_state(self):
        """load_plugin returns None when the plugin is in FAILED state."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class AlwaysFails:
            name = "always-fails"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register(self, registry):
                msg = "always fails"
                raise RuntimeError(msg)

        # Register directly (load only works with entry points)
        pm.register(AlwaysFails())
        assert pm.get_plugin_state("always-fails").value == "failed"

        # load_plugin on a FAILED plugin returns None
        result = pm.load_plugin("always-fails")
        assert result is None


# ------------------------------------------------------------------
# N. __getattr__ / hasattr side effects
# ------------------------------------------------------------------


class TestGetAttrSideEffects:
    """
    Verify that attribute access does not leave corrupted state or
    trigger repeated expensive discovery.
    """

    def test_multiple_access_same_reader(self):
        """Accessing the same reader multiple times does not re-discover."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeReaderPlugin()
        pm.register(plugin)

        # Access via __getattr__ (mimics scp.read_fake)
        reader1 = registry.get_reader("fake")
        reader2 = registry.get_reader("fake")
        assert reader1 is reader2

    def test_hasattr_on_unknown_does_not_crash(self):
        """Hasattr on a truly unknown attribute does not leave corrupted state."""
        import spectrochempy as scp

        # Accessing an unknown attribute should not crash hasattr
        # (hasattr internally calls __getattr__ and catches AttributeError)
        result = hasattr(scp, "this_attribute_does_not_exist_xyz")
        assert result is False

    def test_hasattr_on_known_reader(self):
        """Hasattr returns True for a registered reader name."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class SimpleReaderPlugin:
            name = "simple-reader"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return [
                    {
                        "name": "simplereader",
                        "func": lambda p: p,
                    }
                ]

        pm.register(SimpleReaderPlugin())
        assert registry.get_reader("simplereader") is not None


# ------------------------------------------------------------------
# F. Lazy proxy introspection
# ------------------------------------------------------------------


class TestLazyProxyIntrospection:
    """``lazy_proxy`` introspection behaviour (real doc/signature)."""

    def test_nmr_read_topspin_doc(self):
        """__doc__ is the real read_topspin docstring, not a wrapper one."""
        import spectrochempy as scp

        doc = scp.nmr.read_topspin.__doc__
        assert doc is not None
        assert "Lazy wrapper" not in doc

    def test_nmr_read_topspin_name(self):
        """__name__ is 'read_topspin'."""
        import spectrochempy as scp

        assert scp.nmr.read_topspin.__name__ == "read_topspin"

    def test_nmr_read_topspin_wrapped(self):
        """__wrapped__ is the real read_topspin function."""
        import spectrochempy as scp

        wrapped = scp.nmr.read_topspin.__wrapped__
        assert wrapped is not None
        assert wrapped.__name__ == "read_topspin"

    def test_nmr_signature(self):
        """inspect.signature returns real params, not just (*args, **kwargs)."""
        import inspect

        import spectrochempy as scp

        sig = inspect.signature(scp.nmr.read_topspin)
        params = list(sig.parameters.keys())
        assert "paths" in params or len(params) > 1

    @pytest.mark.skipif(not _CANTERA_INSTALLED, reason="cantera plugin not installed")
    def test_cantera_pfr_doc(self):
        """__doc__ is the real PFR docstring."""
        import spectrochempy as scp

        doc = scp.cantera.PFR.__doc__
        assert doc is not None
        assert "Lazy wrapper" not in doc

    @pytest.mark.skipif(not _CANTERA_INSTALLED, reason="cantera plugin not installed")
    def test_cantera_pfr_name(self):
        """__name__ is 'PFR'."""
        import spectrochempy as scp

        assert scp.cantera.PFR.__name__ == "PFR"

    @pytest.mark.skipif(not _CANTERA_INSTALLED, reason="cantera plugin not installed")
    def test_cantera_pfr_wrapped(self):
        """__wrapped__ is the real PFR class."""
        import spectrochempy as scp

        wrapped = scp.cantera.PFR.__wrapped__
        assert wrapped is not None

    @pytest.mark.skipif(not _CANTERA_INSTALLED, reason="cantera plugin not installed")
    def test_cantera_pfr_signature(self):
        """inspect.signature returns real params, not placeholder ones."""
        import inspect

        import spectrochempy as scp

        sig = inspect.signature(scp.cantera.PFR)
        params = list(sig.parameters.keys())
        assert len(params) > 2

    def test_lazy_loading_preserved_on_import(self):
        """Import spectrochempy does NOT eagerly resolve lazy_proxy objects."""
        import subprocess
        import sys

        code = """
import sys
import spectrochempy as scp
scp.plugin_manager.discover()
lazy = [
    "spectrochempy_nmr.readers.read_topspin",
    "spectrochempy_cantera._pfr",
]
for mod in lazy:
    if mod in sys.modules:
        print(f"EAGER on scp only: {mod}")
        raise SystemExit(1)
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_lazy_loading_proxy_creation_no_resolve(self):
        """Accessing scp.nmr.read_topspin returns unresolved proxy."""
        import subprocess
        import sys

        code = """
import sys
import spectrochempy as scp
scp.plugin_manager.discover()
before = [k for k in sys.modules if k.startswith("spectrochempy_nmr.")]
rt = scp.nmr.read_topspin
after = [k for k in sys.modules if k.startswith("spectrochempy_nmr.")]
if "spectrochempy_nmr.readers.read_topspin" in after:
    print(f"RESOLVED on proxy creation: {after}")
    raise SystemExit(1)
_ = rt.__doc__
resolved = [k for k in sys.modules if k.startswith("spectrochempy_nmr.")]
if "spectrochempy_nmr.readers.read_topspin" not in resolved:
    print(f"Not resolved after __doc__: {resolved}")
    raise SystemExit(1)
raise SystemExit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


# ------------------------------------------------------------------
# O. register_handlers integration
# ------------------------------------------------------------------


class TestRegisterHandlers:
    """Integration tests for the register_handlers declarative hook."""

    def test_handler_collection(self):
        """register_handlers is collected when a plugin is registered."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        def my_handler():
            return "handled"

        class HandlerPlugin(SpectroChemPyPlugin):
            name = "handler_plugin"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_handlers(self):
                return {"test.handler": my_handler}

        pm.register(HandlerPlugin())
        assert registry.get_handler("test.handler")() == "handled"

    def test_handler_collection_returns_none(self):
        """A plugin that returns None from register_handlers is handled gracefully."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class NoneHandlerPlugin(SpectroChemPyPlugin):
            name = "none_handler"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_handlers(self):
                return None

        pm.register(NoneHandlerPlugin())
        assert registry.available_handlers == {}

    def test_handler_collection_returns_empty_dict(self):
        """A plugin that returns {} from register_handlers registers nothing."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        class EmptyHandlerPlugin(SpectroChemPyPlugin):
            name = "empty_handler"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_handlers(self):
                return {}

        pm.register(EmptyHandlerPlugin())
        assert registry.available_handlers == {}

    def test_handler_composable_first_non_none_wins(self):
        """When two plugins register the same handler name, first non-None wins."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)

        def handler_a():
            return "a"

        def handler_b():
            return "b"

        class PluginA(SpectroChemPyPlugin):
            name = "plugin_a"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_handlers(self):
                return {"dup.handler": handler_a}

        class PluginB(SpectroChemPyPlugin):
            name = "plugin_b"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_handlers(self):
                return {"dup.handler": handler_b}

        pm.register(PluginA())
        pm.register(PluginB())
        assert registry.get_handler("dup.handler")() == "a"
        assert registry.available_handlers["dup.handler"] == [handler_a, handler_b]

    def test_coord_reversed_no_handler_uses_default(self):
        """Without any coord.reversed handler, ppm and 1/centimeter still reverse."""
        from spectrochempy.core.dataset.coord import Coord

        c = Coord([1, 2, 3], units="ppm")
        assert c.reversed is True

    def test_coord_reversed_no_handler_shows_false(self):
        """Without any coord.reversed handler, meter stays unreversed."""
        from spectrochempy.core.dataset.coord import Coord

        c = Coord([1, 2, 3], units="meter")
        assert c.reversed is False

    def test_coord_reversed_custom_handler_overrides_default(self, monkeypatch):
        """
        A coord.reversed handler that returns a non-None value takes
        precedence over the default ppm-reversal logic.
        """
        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.plugins import manager as manager_module

        def always_false(coord):
            return False

        monkeypatch.setattr(
            manager_module.plugin_manager.registry,
            "get_handler",
            lambda name: always_false if name == "coord.reversed" else None,
        )
        c = Coord([1, 2, 3], units="ppm")
        assert c.reversed is False

    def test_coord_reversed_handler_none_falls_through(self, monkeypatch):
        """A handler returning None lets default logic run."""
        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.plugins import manager as manager_module

        def maybe_reversed(coord):
            if coord.units == "ppm":
                return None  # let default handle it
            return False

        monkeypatch.setattr(
            manager_module.plugin_manager.registry,
            "get_handler",
            lambda name: maybe_reversed if name == "coord.reversed" else None,
        )
        c = Coord([1, 2, 3], units="ppm")
        assert c.reversed is True
        c2 = Coord([1, 2, 3], units="meter")
        assert c2.reversed is False


# ------------------------------------------------------------------
# P. Plugin test directory layout guard
# ------------------------------------------------------------------


class TestPluginTestDirLayout:
    """
    Guard: plugin test directories must NOT be Python packages.

    If a ``plugins/*/tests/__init__.py`` exists, pytest's import machinery
    treats every plugin test directory as a top-level ``tests`` package,
    causing ``ModuleNotFoundError`` when multiple plugin test directories
    are collected together (e.g. ``tests.test_iris`` collides across
    different plugin test dirs).

    Remove any ``plugins/*/tests/__init__.py`` files instead.
    """

    PLUGIN_TEST_DIRS = sorted(Path("plugins").glob("*/tests"))

    def test_plugin_test_dirs_are_not_packages(self):
        """No ``plugins/*/tests/__init__.py`` exists."""
        offenders = [d for d in self.PLUGIN_TEST_DIRS if (d / "__init__.py").exists()]
        assert not offenders, (
            f"plugin test directories must not be packages (remove __init__.py): "
            f"{[str(p) for p in offenders]}"
        )

    def test_plugin_test_dirs_contain_test_files(self):
        """Each ``plugins/*/tests/`` directory has at least one ``test_*.py``."""
        empty = [d for d in self.PLUGIN_TEST_DIRS if not list(d.glob("test_*.py"))]
        assert (
            not empty
        ), f"plugin test dirs without test files: {[str(p) for p in empty]}"
