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


def test_final_release_minimum_rejects_prereleases(checker):
    constraint = ">=1.1.0,<2"
    assert not checker.version_allowed("1.1.0.dev1", constraint)
    assert not checker.version_allowed("1.1.0rc1", constraint)
    assert checker.version_allowed("1.1.0", constraint)


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


def test_validation_version_preserves_compatible_development_version(checker, tmp_path):
    plugin = tmp_path / "spectrochempy-official"
    _write_pyproject(plugin, official=True, constraint=">=1.1.0,<2")

    assert (
        checker.validation_core_version("1.1.1.dev0", [plugin / "pyproject.toml"])
        == "1.1.1.dev0"
    )


def test_validation_version_uses_final_plugin_floor(checker, tmp_path):
    old_plugin = tmp_path / "spectrochempy-old"
    new_plugin = tmp_path / "spectrochempy-new"
    _write_pyproject(old_plugin, official=True, constraint=">=0.9.0,<2")
    _write_pyproject(new_plugin, official=True, constraint=">=1.1.0,<2")

    assert (
        checker.validation_core_version(
            "1.0.1.dev0",
            [old_plugin / "pyproject.toml", new_plugin / "pyproject.toml"],
        )
        == "1.1.0"
    )


def test_validation_version_rejects_incompatible_bounds(checker, tmp_path):
    first = tmp_path / "spectrochempy-first"
    second = tmp_path / "spectrochempy-second"
    _write_pyproject(first, official=True, constraint=">=1.1.0,<1.2")
    _write_pyproject(second, official=True, constraint=">=1.2.0,<2")

    with pytest.raises(ValueError, match="incompatible with"):
        checker.validation_core_version(
            "1.0.1.dev0",
            [first / "pyproject.toml", second / "pyproject.toml"],
        )


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


@pytest.mark.parametrize(
    ("plugin", "import_name"),
    [
        ("spectrochempy-nmr", "spectrochempy_nmr"),
        ("spectrochempy-perkinelmer", "spectrochempy_perkinelmer"),
        ("spectrochempy-iris", "spectrochempy_iris"),
        ("spectrochempy-tensor", "spectrochempy_tensor"),
    ],
)
def test_1_1_plugin_constraints_are_aligned(checker, plugin, import_name):
    repo_root = Path(__file__).parents[3]
    plugin_dir = repo_root / "plugins" / plugin

    assert (
        checker.read_spectrochempy_constraint(plugin_dir / "pyproject.toml")
        == ">=1.1.0,<2"
    )
    assert "spectrochempy >=1.1.0,<2" in (plugin_dir / "recipe.yaml").read_text()

    init_text = (plugin_dir / "src" / import_name / "__init__.py").read_text()
    assert 'spectrochempy_min_version = "1.1.0"' in init_text
