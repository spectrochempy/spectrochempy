from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".github"
    / "workflows"
    / "scripts"
    / "validate_official_plugin.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_official_plugin", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_plugin_pyproject(
    tmp_path: Path,
    *,
    official_marker: bool = True,
    keep_legacy_classifier: bool = False,
) -> Path:
    """Write a minimal pyproject.toml and return the plugin directory."""
    plugin_dir = tmp_path / "spectrochempy-nmr"
    plugin_dir.mkdir()
    classifiers = []
    if keep_legacy_classifier:
        classifiers.append("Framework :: SpectroChemPy :: Official Plugin")

    classifiers_str = (
        ("\nclassifiers = " + repr(classifiers)) if classifiers else ""
    )
    tool_section = (
        "\n[tool.spectrochempy]\nofficial-plugin = true" if official_marker else ""
    )
    (plugin_dir / "pyproject.toml").write_text(
        f'[project]\nname = "spectrochempy-nmr"\nversion = "0.1.0"'
        f"{classifiers_str}\n{tool_section}\n"
    )
    return plugin_dir


class TestValidatePlugin:
    def test_valid_official_plugin(self, tmp_path):
        module = load_module()
        plugin_dir = _write_plugin_pyproject(tmp_path, official_marker=True)

        errors = module.validate_plugin(plugin_dir)

        assert errors == []

    def test_missing_official_marker(self, tmp_path):
        module = load_module()
        plugin_dir = _write_plugin_pyproject(tmp_path, official_marker=False)

        errors = module.validate_plugin(plugin_dir)

        assert len(errors) == 1
        assert "official-plugin is not set to true" in errors[0]

    def test_legacy_classifier_triggers_error(self, tmp_path):
        module = load_module()
        plugin_dir = _write_plugin_pyproject(
            tmp_path, official_marker=True, keep_legacy_classifier=True
        )

        errors = module.validate_plugin(plugin_dir)

        assert len(errors) == 1
        assert "invalid Trove classifier" in errors[0]

    def test_both_issues(self, tmp_path):
        module = load_module()
        plugin_dir = _write_plugin_pyproject(
            tmp_path, official_marker=False, keep_legacy_classifier=True
        )

        errors = module.validate_plugin(plugin_dir)

        assert len(errors) == 2

    def test_missing_pyproject(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()

        errors = module.validate_plugin(plugin_dir)

        assert len(errors) == 1
        assert "pyproject.toml not found" in errors[0]


class TestMain:
    def test_main_valid(self, tmp_path):
        module = load_module()
        plugin_dir = _write_plugin_pyproject(tmp_path)

        assert module.main.__code__ is not None  # smoke test for import
        # Test via validate_plugin directly since main() uses sys.argv
        errors = module.validate_plugin(plugin_dir)
        assert errors == []

    def test_main_no_args(self, monkeypatch):
        module = load_module()
        monkeypatch.setattr(sys, "argv", ["validate_official_plugin.py"])

        assert module.main() == 2
