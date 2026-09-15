"""Tests for .github/workflows/scripts/check_plugin_core_compatibility.py."""

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
    / "check_plugin_core_compatibility.py"
)

packaging = pytest.importorskip("packaging")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "check_plugin_core_compatibility", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def checker():
    return load_module()


def _write_pyproject(
    plugin_dir: Path, *, official: bool, constraint: str = ">=0.12,<2"
) -> None:
    plugin_dir.mkdir()
    tool = ""
    if official:
        tool = "\n[tool.spectrochempy]\nofficial-plugin = true\n"
    (plugin_dir / "pyproject.toml").write_text(
        f'[project]\nname = "{plugin_dir.name}"\n'
        f'version = "0.1.0"\ndependencies = [\n'
        f'    "spectrochempy{constraint}",\n]\n{tool}'
    )


def test_version_allowed_accepts_release_candidates(checker):
    assert checker.version_allowed("1.0.0rc1", ">=0.12,<2")
    assert checker.version_allowed("1.0.0rc2", ">=0.12,<2")
    assert checker.version_allowed("1.0.0", ">=0.12,<2")


def test_version_allowed_accepts_late_012(checker):
    assert checker.version_allowed("0.12.8", ">=0.12,<2")


def test_version_allowed_rejects_next_major(checker):
    assert not checker.version_allowed("2.0.0", ">=0.12,<2")
    assert not checker.version_allowed("2.0.0rc1", ">=0.12,<2")


def test_version_allowed_rejects_below_minimum(checker):
    assert checker.version_allowed("0.12.0", ">=0.12,<2")
    assert not checker.version_allowed("0.11.9", ">=0.12,<2")


def test_version_allowed_invalid(checker):
    assert not checker.version_allowed("not-a-version", ">=0.12,<2")


def test_is_official_plugin(checker, tmp_path):
    official = tmp_path / "spectrochempy-official"
    non_official = tmp_path / "spectrochempy-non-official"
    _write_pyproject(official, official=True)
    _write_pyproject(non_official, official=False)
    assert checker.is_official_plugin(official / "pyproject.toml")
    assert not checker.is_official_plugin(non_official / "pyproject.toml")


def test_read_spectrochempy_constraint(checker, tmp_path):
    plugin_dir = tmp_path / "spectrochempy-test"
    _write_pyproject(plugin_dir, official=True, constraint=">=0.12,<2")
    assert (
        checker.read_spectrochempy_constraint(plugin_dir / "pyproject.toml")
        == ">=0.12,<2"
    )


def test_read_spectrochempy_constraint_missing(checker, tmp_path):
    plugin_dir = tmp_path / "spectrochempy-nodep"
    plugin_dir.mkdir()
    (plugin_dir / "pyproject.toml").write_text("unrelated = true\n")
    assert checker.read_spectrochempy_constraint(plugin_dir / "pyproject.toml") is None
