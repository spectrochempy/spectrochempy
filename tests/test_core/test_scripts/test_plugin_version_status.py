from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".github"
    / "workflows"
    / "scripts"
    / "plugin_version_status.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("plugin_version_status", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_plugin_tag():
    module = load_module()

    assert module.parse_tag("spectrochempy-nmr-v0.1.3") == (
        "spectrochempy-nmr",
        "0.1.3",
    )
    assert module.parse_tag("spectrochempy-v0.9.2") is None


def test_next_patch_dev_version_is_newer_than_latest_stable():
    module = load_module()

    assert module.next_patch_dev_version("0.1.3", 12) == "0.1.4.dev12"


def test_release_relevant_paths_excludes_tests_and_docs(tmp_path):
    module = load_module()
    plugin_dir = tmp_path / "plugins" / "spectrochempy-nmr"
    (plugin_dir / "src").mkdir(parents=True)
    (plugin_dir / "tests").mkdir()
    (plugin_dir / "docs").mkdir()
    (plugin_dir / "pyproject.toml").write_text("")
    (plugin_dir / "recipe.yaml").write_text("")

    paths = module.release_relevant_paths(plugin_dir)

    assert paths == [
        (plugin_dir / "src").as_posix(),
        (plugin_dir / "pyproject.toml").as_posix(),
        (plugin_dir / "recipe.yaml").as_posix(),
    ]


def test_apply_dev_version_updates_plugin_metadata(tmp_path, monkeypatch):
    module = load_module()
    plugin_dir = tmp_path / "plugins" / "spectrochempy-nmr"
    init_dir = plugin_dir / "src" / "spectrochempy_nmr"
    init_dir.mkdir(parents=True)
    (plugin_dir / "pyproject.toml").write_text(
        '[project]\nname = "spectrochempy-nmr"\nversion = "0.1.3"\n'
    )
    (plugin_dir / "recipe.yaml").write_text(
        'context:\n  name: spectrochempy-nmr\n  version: "0.1.3"\n'
    )
    (init_dir / "__init__.py").write_text('class NMRPlugin:\n    version = "0.1.3"\n')
    monkeypatch.chdir(tmp_path)
    status = module.PluginVersionStatus(
        plugin="spectrochempy-nmr",
        plugin_dir="plugins/spectrochempy-nmr",
        latest_tag="spectrochempy-nmr-v0.1.3",
        base_version="0.1.3",
        commits_since_tag=12,
        dev_version="0.1.4.dev12",
        changed_files=3,
        has_changes=True,
    )

    changed_paths = module.apply_dev_version(status)

    assert changed_paths == [
        "plugins/spectrochempy-nmr/pyproject.toml",
        "plugins/spectrochempy-nmr/recipe.yaml",
        "plugins/spectrochempy-nmr/src/spectrochempy_nmr/__init__.py",
    ]
    assert 'version = "0.1.4.dev12"' in (plugin_dir / "pyproject.toml").read_text()
    assert 'version: "0.1.4.dev12"' in (plugin_dir / "recipe.yaml").read_text()
    assert 'version = "0.1.4.dev12"' in (init_dir / "__init__.py").read_text()


def test_discover_official_plugins_with_marker(tmp_path, monkeypatch):
    """Verify _discover_official_plugins detects the [tool.spectrochempy] marker."""
    module = load_module()

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # Official plugin: has official-plugin = true
    official = plugins_dir / "spectrochempy-nmr"
    official.mkdir()
    (official / "pyproject.toml").write_text(
        '[project]\nname = "spectrochempy-nmr"\n'
        '[tool.spectrochempy]\nofficial-plugin = true\n'
    )

    # Non-official plugin: no marker
    non_official = plugins_dir / "spectrochempy-cantera"
    non_official.mkdir()
    (non_official / "pyproject.toml").write_text(
        '[project]\nname = "spectrochempy-cantera"\n'
    )

    monkeypatch.chdir(tmp_path)
    result = module._discover_official_plugins()

    assert result == ("spectrochempy-nmr",)


def test_discover_official_plugins_empty(tmp_path, monkeypatch):
    """Verify _discover_official_plugins returns empty tuple when no plugins dir."""
    module = load_module()

    monkeypatch.chdir(tmp_path)
    result = module._discover_official_plugins()

    assert result == ()
