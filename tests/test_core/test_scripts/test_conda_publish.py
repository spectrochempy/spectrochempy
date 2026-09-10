"""Tests for .github/workflows/scripts/conda_publish.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".github"
    / "workflows"
    / "scripts"
    / "conda_publish.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("conda_publish", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------


class TestParsePluginTag:
    def test_valid_tag(self):
        module = load_module()
        result = module.parse_plugin_tag("spectrochempy-nmr-v0.1.11")
        assert result == ("spectrochempy-nmr", "0.1.11")

    def test_valid_tag_carroucell(self):
        module = load_module()
        result = module.parse_plugin_tag("spectrochempy-carroucell-v0.1.7")
        assert result == ("spectrochempy-carroucell", "0.1.7")

    def test_core_tag_returns_none(self):
        module = load_module()
        assert module.parse_plugin_tag("spectrochempy-v0.12.0") is None

    def test_invalid_format_returns_none(self):
        module = load_module()
        assert module.parse_plugin_tag("v0.1.0") is None
        assert module.parse_plugin_tag("spectrochempy-nmr") is None
        assert module.parse_plugin_tag("random-tag") is None

    def test_version_with_patch(self):
        module = load_module()
        result = module.parse_plugin_tag("spectrochempy-iris-v1.2.3")
        assert result == ("spectrochempy-iris", "1.2.3")


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------


class TestValidateTagFormat:
    def test_valid_tag(self):
        module = load_module()
        plugin, version = module.validate_tag_format("spectrochempy-nmr-v0.1.11")
        assert plugin == "spectrochempy-nmr"
        assert version == "0.1.11"

    def test_valid_tag_with_expected_plugin(self):
        module = load_module()
        plugin, version = module.validate_tag_format(
            "spectrochempy-nmr-v0.1.11", expected_plugin="spectrochempy-nmr"
        )
        assert plugin == "spectrochempy-nmr"

    def test_wrong_plugin_raises(self):
        module = load_module()
        with pytest.raises(ValueError, match="belongs to plugin"):
            module.validate_tag_format(
                "spectrochempy-nmr-v0.1.11", expected_plugin="spectrochempy-iris"
            )

    def test_invalid_tag_raises(self):
        module = load_module()
        with pytest.raises(ValueError, match="does not match"):
            module.validate_tag_format("spectrochempy-v0.12.0")


# ---------------------------------------------------------------------------
# Plugin version reading
# ---------------------------------------------------------------------------


class TestReadPluginVersion:
    def test_read_from_pyproject(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-nmr"\nversion = "0.1.11"\n'
        )
        assert module.read_plugin_version(plugin_dir) == "0.1.11"

    def test_read_from_init(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-nmr"
        init_dir = plugin_dir / "src" / "spectrochempy_nmr"
        init_dir.mkdir(parents=True)
        (init_dir / "__init__.py").write_text('    version = "0.1.11"\n')
        assert module.read_plugin_init_version(plugin_dir) == "0.1.11"

    def test_read_recipe_version(self, tmp_path):
        module = load_module()
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(
            'context:\n  name: spectrochempy-nmr\n  version: "0.1.11"\n'
        )
        assert module.read_recipe_version(recipe) == "0.1.11"

    def test_missing_pyproject(self, tmp_path):
        module = load_module()
        assert module.read_plugin_version(tmp_path / "nonexistent") is None

    def test_missing_init(self, tmp_path):
        module = load_module()
        assert module.read_plugin_init_version(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# Official plugin discovery
# ---------------------------------------------------------------------------


class TestDiscoverOfficialPlugins:
    def test_finds_official_plugins(self, tmp_path, monkeypatch):
        module = load_module()
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        # Official plugin
        official = plugins_dir / "spectrochempy-nmr"
        official.mkdir()
        (official / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-nmr"\n'
            "[tool.spectrochempy]\nofficial-plugin = true\n"
        )

        # Non-official plugin
        non_official = plugins_dir / "spectrochempy-cantera"
        non_official.mkdir()
        (non_official / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-cantera"\n'
        )

        # Plugin template (excluded)
        template = plugins_dir / "plugin-template"
        template.mkdir()
        (template / "pyproject.toml").write_text(
            '[project]\nname = "plugin-template"\n'
            "[tool.spectrochempy]\nofficial-plugin = true\n"
        )

        monkeypatch.chdir(tmp_path)
        result = module.discover_official_plugins(plugins_dir)
        assert result == ["spectrochempy-nmr"]

    def test_empty_dir(self, tmp_path, monkeypatch):
        module = load_module()
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        assert module.discover_official_plugins(plugins_dir) == []

    def test_nonexistent_dir(self):
        module = load_module()
        assert module.discover_official_plugins(Path("/nonexistent")) == []


# ---------------------------------------------------------------------------
# is_official_plugin
# ---------------------------------------------------------------------------


class TestIsOfficialPlugin:
    def test_official(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            "[tool.spectrochempy]\nofficial-plugin = true\n"
        )
        assert module.is_official_plugin(plugin_dir) is True

    def test_not_official(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-cantera"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-cantera"\n'
        )
        assert module.is_official_plugin(plugin_dir) is False


class TestIsOfficialCli:
    """Exercise the `is-official` CLI as the discovery job does."""

    def _run(self, *args):
        import subprocess

        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "is-official", *args],
            capture_output=True,
            text=True,
        )

    def test_official_exit_zero(self, tmp_path):
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            "[tool.spectrochempy]\nofficial-plugin = true\n"
        )
        result = self._run(str(plugin_dir))
        assert result.returncode == 0

    def test_not_official_exit_one(self, tmp_path):
        plugin_dir = tmp_path / "spectrochempy-cantera"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-cantera"\n'
        )
        result = self._run(str(plugin_dir))
        assert result.returncode == 1

    def test_missing_pyproject_exit_one(self, tmp_path):
        result = self._run(str(tmp_path))
        assert result.returncode == 1

    def test_parse_error_exit_one(self, tmp_path):
        """Malformed TOML is treated as 'not official' by the module."""
        plugin_dir = tmp_path / "spectrochempy-cantera"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("[tool.spectrochempy\n")
        result = self._run(str(plugin_dir))
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Release consistency (network-independent)
# ---------------------------------------------------------------------------


class TestCheckPluginReleaseConsistency:
    def test_aligned_local_check(self, tmp_path, monkeypatch):
        """Simulate a release that exists on GitHub (tag present) with skip_network."""
        module = load_module()
        monkeypatch.chdir(tmp_path)

        # Create a fake git repo with the tag
        import subprocess

        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("init")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "tag", "spectrochempy-nmr-v0.1.11"],
            check=True,
            capture_output=True,
        )

        check = module.check_plugin_release_consistency(
            "spectrochempy-nmr", "0.1.11", skip_network=True
        )
        assert check.github_release is True
        assert check.verdict == "conda_missing"

    def test_missing_github_release(self, tmp_path, monkeypatch):
        """Simulate a release with no GitHub tag."""
        module = load_module()
        monkeypatch.chdir(tmp_path)

        import subprocess

        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("init")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], check=True, capture_output=True
        )

        check = module.check_plugin_release_consistency(
            "spectrochempy-nmr", "0.1.11", skip_network=True
        )
        assert check.github_release is False
        assert check.verdict == "missing_github_release"


class TestAnacondaVersions:
    def _plugin(self, monkeypatch):
        module = load_module()
        monkeypatch.setattr(module, "fetch_json", lambda url, timeout=30: None)
        return module

    def test_str_labels(self, monkeypatch):
        module = load_module()

        def fake_fetch(url, timeout=30):
            return [
                {"version": "0.1.0", "labels": ["dev"]},
                {"version": "0.1.1", "labels": ["dev", "main"]},
                {"version": "0.1.1", "labels": ["main"]},
            ]

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        assert module.anaconda_versions("spectrochempy-nmr") == {
            "0.1.0": ["dev"],
            "0.1.1": ["dev", "main"],
        }

    def test_dict_labels(self, monkeypatch):
        module = load_module()
        monkeypatch.setattr(
            module,
            "fetch_json",
            lambda url, timeout=30: [
                {"version": "0.1.2", "labels": [{"name": "main"}]}
            ],
        )
        assert module.anaconda_versions("spectrochempy-nmr") == {
            "0.1.2": ["main"]
        }

    def test_package_not_found(self, monkeypatch):
        module = self._plugin(monkeypatch)
        assert module.anaconda_versions("spectrochempy-perkinelmer") == {}

    def test_version_labels_uses_versions_map(self, monkeypatch):
        module = load_module()
        monkeypatch.setattr(
            module,
            "fetch_json",
            lambda url, timeout=30: [
                {"version": "0.1.1", "labels": ["dev", "main"]}
            ],
        )
        assert module.anaconda_version_labels("spectrochempy-nmr", "0.1.1") == [
            "dev",
            "main",
        ]
        assert module.anaconda_version_labels("spectrochempy-nmr", "0.9.9") == []


class TestCheckPluginReleaseConsistencyNetwork:
    def _make_repo(self, tmp_path, tag):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("init")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(["git", "tag", tag], cwd=tmp_path, check=True, capture_output=True)

    def test_aligned(self, tmp_path, monkeypatch):
        module = load_module()
        self._make_repo(tmp_path, "spectrochempy-nmr-v0.1.11")
        monkeypatch.chdir(tmp_path)

        def fake_fetch(url, timeout=30):
            if "pypi.org" in url:
                return {"releases": {"0.1.11": [], "0.1.8": []}}
            if "anaconda.org" in url:
                return [{"version": "0.1.11", "labels": ["main"]}]
            return None

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        check = module.check_plugin_release_consistency("spectrochempy-nmr", "0.1.11")
        assert check.verdict == "aligned"
        assert check.pypi_version == "0.1.11"
        assert check.conda_main is True

    def test_pypi_missing(self, tmp_path, monkeypatch):
        module = load_module()
        self._make_repo(tmp_path, "spectrochempy-nmr-v0.1.11")
        monkeypatch.chdir(tmp_path)

        def fake_fetch(url, timeout=30):
            if "pypi.org" in url:
                return {"releases": {"0.1.8": []}}
            if "anaconda.org" in url:
                return [{"version": "0.1.11", "labels": ["main"]}]
            return None

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        check = module.check_plugin_release_consistency("spectrochempy-nmr", "0.1.11")
        assert check.verdict == "pypi_missing"
        assert check.pypi_version is None

    def test_conda_dev_only(self, tmp_path, monkeypatch):
        module = load_module()
        self._make_repo(tmp_path, "spectrochempy-nmr-v0.1.11")
        monkeypatch.chdir(tmp_path)

        def fake_fetch(url, timeout=30):
            if "pypi.org" in url:
                return {"releases": {"0.1.11": []}}
            if "anaconda.org" in url:
                return [{"version": "0.1.11", "labels": ["dev"]}]
            return None

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        check = module.check_plugin_release_consistency("spectrochempy-nmr", "0.1.11")
        assert check.verdict == "conda_dev_only"
        assert check.conda_dev is True


# ---------------------------------------------------------------------------
# Format verification report
# ---------------------------------------------------------------------------


class TestFormatVerificationReport:
    def test_aligned_report(self):
        module = load_module()
        check = module.PluginReleaseCheck(
            plugin="spectrochempy-nmr",
            version="0.1.11",
            github_release=True,
            pypi_version="0.1.11",
            conda_main=True,
            conda_dev=False,
            conda_labels=["main"],
            verdict="aligned",
        )
        report = module.format_verification_report([check])
        assert "spectrochempy-nmr" in report
        assert "0.1.11" in report
        assert "aligned" in report

    def test_missing_conda_report(self):
        module = load_module()
        check = module.PluginReleaseCheck(
            plugin="spectrochempy-carroucell",
            version="0.1.7",
            github_release=True,
            pypi_version="0.1.7",
            conda_main=False,
            conda_dev=False,
            conda_labels=[],
            verdict="conda_missing",
        )
        report = module.format_verification_report([check])
        assert "stable Conda missing" in report


# ---------------------------------------------------------------------------
# No-upload protection
# ---------------------------------------------------------------------------


class TestNoUploadProtection:
    """Ensure test environment cannot trigger real uploads."""

    def test_upload_function_not_called_in_tests(self):
        """Verify anaconda upload is never executed during test runs."""
        module = load_module()
        # The module should not have any subprocess call to anaconda upload
        # at import time or during pure logic functions.
        source = SCRIPT_PATH.read_text()
        # The upload command should only appear inside workflow shell scripts
        # (in the YAML files), not in the Python module's functions.
        # This is a structural check — the Python module uses urllib for reads
        # and never calls anaconda CLI directly.
        assert "subprocess.run" in source  # git commands use subprocess
        # But no anaconda upload calls
        assert "anaconda upload" not in source

    def test_fetch_json_uses_read_only_urls(self):
        """Verify fetch_json only accesses read-only API endpoints."""
        module = load_module()
        # The URLs in the module should be read-only API endpoints
        assert "pypi.org/pypi" in module.PYPI_JSON_URL
        assert "api.anaconda.org" in module.ANACONDA_FILES_URL


# ---------------------------------------------------------------------------
# Tag regex patterns
# ---------------------------------------------------------------------------


class TestTagRegex:
    def test_plugin_tag_pattern(self):
        module = load_module()
        assert module.TAG_RE.match("spectrochempy-nmr-v0.1.11") is not None
        assert module.TAG_RE.match("spectrochempy-carroucell-v0.1.7") is not None
        assert module.TAG_RE.match("spectrochempy-tensor-v0.1.5") is not None

    def test_core_tag_does_not_match(self):
        module = load_module()
        assert module.TAG_RE.match("spectrochempy-v0.12.0") is None

    def test_invalid_patterns(self):
        module = load_module()
        assert module.TAG_RE.match("v0.1.0") is None
        assert module.TAG_RE.match("spectrochempy-nmr") is None
        assert module.TAG_RE.match("spectrochempy-nmr-v") is None
        assert module.TAG_RE.match("spectrochempy-nmr-vabc") is None


# ---------------------------------------------------------------------------
# Concurrency key behavior (structural test)
# ---------------------------------------------------------------------------


class TestConcurrencyKeyBehavior:
    """Verify the workflow YAML uses per-tag concurrency that doesn't cancel stable builds."""

    def test_build_package_yml_concurrency(self):
        workflow = (
            Path(__file__).parents[3]
            / ".github"
            / "workflows"
            / "build_package.yml"
        ).read_text()
        # Should NOT cancel in-progress for stable releases
        assert "cancel-in-progress:" in workflow
        # Should have conditional cancel logic
        assert "startsWith(github.ref_name, 'spectrochempy-v')" in workflow

    def test_no_unconditional_cancel(self):
        workflow = (
            Path(__file__).parents[3]
            / ".github"
            / "workflows"
            / "build_package.yml"
        ).read_text()
        # The cancel-in-progress should not be simply "true"
        lines = [
            l.strip()
            for l in workflow.splitlines()
            if "cancel-in-progress" in l
        ]
        assert len(lines) >= 1
        for line in lines:
            assert line != "cancel-in-progress: true"
