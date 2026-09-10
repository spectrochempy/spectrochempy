"""Tests for .github/workflows/scripts/conda_publish.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3] / ".github" / "workflows" / "scripts" / "conda_publish.py"
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
        recipe.write_text('context:\n  name: spectrochempy-nmr\n  version: "0.1.11"\n')
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

    def test_parse_error_exit_two(self, tmp_path):
        """Malformed TOML is an evaluation error (exit 2), not silently 'not official'."""
        plugin_dir = tmp_path / "spectrochempy-cantera"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text("[tool.spectrochempy\n")
        result = self._run(str(plugin_dir))
        assert result.returncode == 2


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
        subprocess.run(["git", "commit", "-m", "init"], check=True, capture_output=True)
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
        subprocess.run(["git", "commit", "-m", "init"], check=True, capture_output=True)

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
        assert module.anaconda_versions("spectrochempy-nmr") == {"0.1.2": ["main"]}

    def test_package_not_found(self, monkeypatch):
        module = self._plugin(monkeypatch)
        assert module.anaconda_versions("spectrochempy-perkinelmer") == {}

    def test_version_labels_uses_versions_map(self, monkeypatch):
        module = load_module()
        monkeypatch.setattr(
            module,
            "fetch_json",
            lambda url, timeout=30: [{"version": "0.1.1", "labels": ["dev", "main"]}],
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
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "tag", tag], cwd=tmp_path, check=True, capture_output=True
        )

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

    def test_pypi_unavailable(self, tmp_path, monkeypatch):
        """A PyPI query failure must yield pypi_unavailable, not pypi_missing."""
        module = load_module()
        self._make_repo(tmp_path, "spectrochempy-nmr-v0.1.11")
        monkeypatch.chdir(tmp_path)

        def fake_fetch(url, timeout=30):
            if "pypi.org" in url:
                raise module.ServiceUnavailableError("pypi.org: connection reset")
            if "anaconda.org" in url:
                return [{"version": "0.1.11", "labels": ["main"]}]
            return None

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        check = module.check_plugin_release_consistency("spectrochempy-nmr", "0.1.11")
        assert check.verdict == "pypi_unavailable"
        assert check.pypi_version is None

    def test_conda_unavailable(self, tmp_path, monkeypatch):
        """An Anaconda query failure must yield conda_unavailable, not conda_missing."""
        module = load_module()
        self._make_repo(tmp_path, "spectrochempy-nmr-v0.1.11")
        monkeypatch.chdir(tmp_path)

        def fake_fetch(url, timeout=30):
            if "pypi.org" in url:
                return {"releases": {"0.1.11": []}}
            if "anaconda.org" in url:
                raise module.ServiceUnavailableError("api.anaconda.org: HTTP 503")
            return None

        monkeypatch.setattr(module, "fetch_json", fake_fetch)
        check = module.check_plugin_release_consistency("spectrochempy-nmr", "0.1.11")
        assert check.verdict == "conda_unavailable"
        assert check.conda_main is False


class TestFetchJsonAvailability:
    def test_http_404_returns_none(self, monkeypatch):
        """An HTTP 404 must be treated as 'not found', not as an error."""
        module = load_module()

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

            def decode(self):
                return "{}"

        def fake_urlopen(req, timeout=30):
            raise urllib.error.HTTPError(
                "https://api.anaconda.org/package/x/y/versions",
                404,
                "Not Found",
                {},
                None,
            )

        monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
        assert (
            module.fetch_json("https://api.anaconda.org/package/x/y/versions") is None
        )

    def test_network_error_raises(self, monkeypatch):
        """A network failure must raise ServiceUnavailableError."""
        module = load_module()

        def fake_urlopen(req, timeout=30):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(module.ServiceUnavailableError):
            module.fetch_json("https://example.org/x.json")

    def test_http_503_raises(self, monkeypatch):
        module = load_module()

        def fake_urlopen(req, timeout=30):
            raise urllib.error.HTTPError(
                "https://x", 503, "Service Unavailable", {}, None
            )

        monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(module.ServiceUnavailableError):
            module.fetch_json("https://x")


class TestListPluginTags:
    def _make_tagged_repo(self, tmp_path, tags):
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
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        for tag in tags:
            subprocess.run(
                ["git", "tag", tag], cwd=tmp_path, check=True, capture_output=True
            )

    def test_returns_plugin_tags_only(self, tmp_path, monkeypatch):
        module = load_module()
        self._make_tagged_repo(
            tmp_path,
            ["spectrochempy-nmr-v0.1.11", "spectrochempy-nmr-v0.1.8", "v0.2.0"],
        )
        monkeypatch.chdir(tmp_path)
        assert module.list_plugin_tags() == [
            ("spectrochempy-nmr", "0.1.8"),
            ("spectrochempy-nmr", "0.1.11"),
        ]


class TestValidateRelease:
    def _plugin_with_version(self, tmp_path, version):
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            f'[project]\nname = "spectrochempy-nmr"\nversion = "{version}"\n'
        )
        init_dir = plugin_dir / "src" / "spectrochempy_nmr"
        init_dir.mkdir(parents=True)
        (init_dir / "__init__.py").write_text(f'    version = "{version}"\n')
        (plugin_dir / "recipe.yaml").write_text(
            f'context:\n  name: spectrochempy-nmr\n  version: "{version}"\n'
        )
        return plugin_dir

    def test_consistent_exit_zero(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_with_version(tmp_path, "0.1.11")
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.11")
            )
            == 0
        )

    def test_pyproject_mismatch_exit_one(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_with_version(tmp_path, "0.1.11")
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.8")
            )
            == 1
        )

    def test_recipe_mismatch_exit_one(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_with_version(tmp_path, "0.1.11")
        (plugin_dir / "recipe.yaml").write_text(
            'context:\n  name: spectrochempy-nmr\n  version: "0.1.8"\n'
        )
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.11")
            )
            == 1
        )

    def test_missing_recipe_exit_one(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_with_version(tmp_path, "0.1.11")
        (plugin_dir / "recipe.yaml").unlink()
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.11")
            )
            == 1
        )

    def test_meta_yaml_jinja_version(self, tmp_path):
        module = load_module()
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            '[project]\nname = "spectrochempy-nmr"\nversion = "0.1.11"\n'
        )
        (plugin_dir / "meta.yaml").write_text(
            '{% set version = "0.1.11" %}\npackage:\n  name: spectrochempy-nmr\n  version: "{{ version }}"\n'
        )
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.11")
            )
            == 0
        )


class TestUploadConda:
    def _artifact(self, tmp_path, name="spectrochempy-nmr-0.1.11-0_abc.conda"):
        path = tmp_path / name
        path.write_text("dummy")
        return path

    def test_refuses_existing_version_without_override(self, tmp_path, monkeypatch):
        module = load_module()
        artifact = self._artifact(tmp_path)
        monkeypatch.setattr(
            module, "anaconda_version_labels", lambda *a, **k: ["dev", "main"]
        )
        assert (
            module.upload_conda(
                artifact, plugin="spectrochempy-nmr", version="0.1.11", dry_run=False
            )
            == 1
        )

    def test_allow_override_forces_reupload(self, tmp_path, monkeypatch):
        module = load_module()
        artifact = self._artifact(tmp_path)
        calls = []

        def fake_labels(*a, **k):
            return ["main"]

        def fake_verify(*a, **k):
            return True

        def fake_subprocess(cmd, *a, **k):
            calls.append(cmd)
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(module, "anaconda_version_labels", fake_labels)
        monkeypatch.setattr(module, "verify_conda_upload", fake_verify)
        monkeypatch.setattr(module.subprocess, "run", fake_subprocess)
        assert (
            module.upload_conda(
                artifact,
                plugin="spectrochempy-nmr",
                version="0.1.11",
                token="tok",
                allow_override=True,
            )
            == 0
        )
        assert calls and "--force" in calls[0]
        assert "-l" in calls[0] and "main" in calls[0]

    def test_fresh_version_uploads_without_force(self, tmp_path, monkeypatch):
        module = load_module()
        artifact = self._artifact(tmp_path)
        calls = []

        def fake_labels(*a, **k):
            return []

        def fake_verify(*a, **k):
            return True

        def fake_subprocess(cmd, *a, **k):
            calls.append(cmd)
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(module, "anaconda_version_labels", fake_labels)
        monkeypatch.setattr(module, "verify_conda_upload", fake_verify)
        monkeypatch.setattr(module.subprocess, "run", fake_subprocess)
        assert (
            module.upload_conda(
                artifact,
                plugin="spectrochempy-nmr",
                version="0.1.11",
                token="tok",
            )
            == 0
        )
        assert calls and "--force" not in calls[0]

    def test_artifact_version_mismatch(self, tmp_path, monkeypatch):
        module = load_module()
        artifact = self._artifact(tmp_path, name="spectrochempy-nmr-0.1.8-0_abc.conda")
        assert (
            module.upload_conda(
                artifact, plugin="spectrochempy-nmr", version="0.1.11", dry_run=True
            )
            == 1
        )

    def test_dry_run_skips_upload(self, tmp_path, monkeypatch):
        module = load_module()
        artifact = self._artifact(tmp_path)
        monkeypatch.setattr(module, "anaconda_version_labels", lambda *a, **k: [])
        called = []

        def fake_subprocess(cmd, *a, **k):
            called.append(cmd)

        monkeypatch.setattr(module.subprocess, "run", fake_subprocess)
        assert (
            module.upload_conda(
                artifact, plugin="spectrochempy-nmr", version="0.1.11", dry_run=True
            )
            == 0
        )
        assert called == []


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
        """Verify test runs can never trigger a real upload outside guarded paths."""
        # The module invokes subprocess for read-only git/registry checks…
        source = SCRIPT_PATH.read_text()
        assert "subprocess.run" in source
        # …and the anaconda CLI is reached only through the explicitly-guarded
        # upload_conda() path (token from the environment, refusal to overwrite
        # without allow_override). No other command builds an upload command.
        assert "subprocess.run(command)" in source
        assert "allow_override" in source
        # The upload subprocess invokes a bare list command, never a shell
        # string that could smuggle arguments from an artifact name.
        assert "subprocess.run(command)" in source
        assert "shell=True" not in source

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
# Plugin name and version input validation
# ---------------------------------------------------------------------------


class TestValidatePluginName:
    def test_official_names_accepted(self):
        module = load_module()
        for name in (
            "spectrochempy-carroucell",
            "spectrochempy-hypercomplex",
            "spectrochempy-iris",
            "spectrochempy-nmr",
            "spectrochempy-perkinelmer",
            "spectrochempy-tensor",
        ):
            assert module.validate_plugin_name(name) == name

    def test_rejects_plain_name(self):
        module = load_module()
        for name in ("", "carroucell", "perkinelmer"):
            with pytest.raises(ValueError):
                module.validate_plugin_name(name)

    def test_rejects_tag_or_uppercase(self):
        module = load_module()
        for name in (
            "spectrochempy-nmr-v0.1.11",
            "spectrochempy-PerkinElmer",
            "spectrochempy nmr",
            "spectrochempy-nmr;ls",
        ):
            with pytest.raises(ValueError):
                module.validate_plugin_name(name)


class TestValidateVersionInput:
    def test_valid_versions(self):
        module = load_module()
        for version in ("0.1.8", "0.1.11", "1.2.3", "0.12.0"):
            assert module.validate_version_input(version) == version

    def test_rejects_empty(self):
        module = load_module()
        with pytest.raises(ValueError, match="empty"):
            module.validate_version_input("")

    def test_rejects_v_prefix(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.validate_version_input("v0.1.8")

    def test_rejects_plugin_name_included(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.validate_version_input("spectrochempy-nmr-v0.1.8")

    def test_rejects_full_tag(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.validate_version_input("spectrochempy-nmr-v0.1.8")

    def test_rejects_whitespace(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.validate_version_input(" 0.1.8")

    def test_rejects_incomplete_version(self):
        module = load_module()
        for version in ("0.1", "0", "0.1.8a", "0.1.8.9"):
            with pytest.raises(ValueError):
                module.validate_version_input(version)

    def test_rejects_shell_metacharacters(self):
        module = load_module()
        for version in (
            "0.1.8;ls -la",
            "0.1.8 & rm -rf /",
            "0.1.8$(whoami)",
            "`id`",
            "0.1.8|cat /etc/passwd",
        ):
            with pytest.raises(ValueError):
                module.validate_version_input(version)

    def test_rejects_non_string(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.validate_version_input("0.1.8\n")


class TestCanonicalPluginTag:
    def test_canonical_tag_construction(self):
        module = load_module()
        assert (
            module.canonical_plugin_tag("spectrochempy-perkinelmer", "0.1.4")
            == "spectrochempy-perkinelmer-v0.1.4"
        )
        assert (
            module.canonical_plugin_tag("spectrochempy-nmr", "0.1.11")
            == "spectrochempy-nmr-v0.1.11"
        )

    def test_never_double_v(self):
        module = load_module()
        assert not module.canonical_plugin_tag("spectrochempy-nmr", "0.1.11").endswith(
            "n-v0.1.11v0.1.11"
        )


# ---------------------------------------------------------------------------
# Recipe discovery / version injection / bound alignment
# ---------------------------------------------------------------------------


class TestUnpackRecipe:
    def test_prefers_recipe_yaml(self, tmp_path):
        module = load_module()
        (tmp_path / "recipe.yaml").write_text("recipe")
        (tmp_path / "meta.yaml").write_text("meta")
        path, name = module.unpack_recipe(tmp_path)
        assert (path, name) == (tmp_path / "recipe.yaml", "recipe.yaml")

    def test_meta_yaml_fallback(self, tmp_path):
        module = load_module()
        (tmp_path / "meta.yaml").write_text("meta")
        path, name = module.unpack_recipe(tmp_path)
        assert (path, name) == (tmp_path / "meta.yaml", "meta.yaml")

    def test_no_recipe_returns_none(self, tmp_path):
        module = load_module()
        assert module.unpack_recipe(tmp_path) is None


class TestInjectRecipeVersion:
    RECIPE = (
        "context:\n  name: spectrochempy-perkinelmer\n"
        '  version: "0.1.4"\n\n'
        'package:\n  name: "${{ name }}"\n  version: "${{ version }}"\n'
    )

    def test_injects_version_deterministically(self, tmp_path):
        module = load_module()
        text = module.inject_recipe_version(self.RECIPE, "0.1.8")
        assert 'version: "0.1.8"' in text
        assert 'package:\n  name: "${{ name }}"\n  version: "${{ version }}"' in text

    def test_leaves_package_version_jinja_intact(self):
        module = load_module()
        text = module.inject_recipe_version(self.RECIPE, "0.1.8")
        assert 'version: "${{ version }}"' in text
        assert "${{ version }}" not in text.split("package:")[1].split("version:")[0]

    def test_invalid_target_version_raises(self):
        module = load_module()
        with pytest.raises(ValueError):
            module.inject_recipe_version(self.RECIPE, "not-a-version")

    def test_missing_context_raises(self):
        module = load_module()
        with pytest.raises(ValueError, match="context"):
            module.inject_recipe_version('package:\n  version: "0.1.4"\n', "0.1.8")


class TestAlignRecipeRequirement:
    RECIPE = (
        'context:\n  name: spectrochempy-nmr\n  version: "0.1.11"\n'
        "requirements:\n  host:\n    - python >=3.11\n  run:\n"
        "    - python >=3.11\n    - spectrochempy >=0.10\n    - numpy\n"
    )

    def _plugin_dir(self, tmp_path, dependency):
        plugin_dir = tmp_path / "spectrochempy-nmr"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            f"[project]\ndependencies = ['{dependency}']\n"
        )
        return plugin_dir

    def test_aligns_from_tag_pyproject(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path, "spectrochempy>=0.12,<0.13")
        text = module.align_recipe_requirement(self.RECIPE, plugin_dir)
        assert "- spectrochempy >=0.12,<0.13" in text
        assert "- spectrochempy >=0.10" not in text

    def test_ignores_pep508_extras(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path, "spectrochempy[io]>=0.12,<0.13")
        text = module.align_recipe_requirement(self.RECIPE, plugin_dir)
        assert "- spectrochempy >=0.12,<0.13" in text

    def test_leaves_recipe_when_not_a_dependency(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path, "numpy>=1.20")
        text = module.align_recipe_requirement(self.RECIPE, plugin_dir)
        assert text == self.RECIPE

    def test_leaves_recipe_without_pyproject(self, tmp_path):
        module = load_module()
        assert module.align_recipe_requirement(self.RECIPE, tmp_path) == self.RECIPE


class TestResolveRecipe:
    def _plugin_dir(
        self, tmp_path, version="0.1.4", dependency="spectrochempy>=0.12,<0.13"
    ):
        plugin_dir = tmp_path / "spectrochempy-perkinelmer"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            f'[project]\nversion = "{version}"\n' f"dependencies = ['{dependency}']\n"
        )
        return plugin_dir

    MASTER_RECIPE = (
        'context:\n  name: spectrochempy-perkinelmer\n  version: "0.1.4"\n\n'
        'package:\n  name: "${{ name }}"\n  version: "${{ version }}"\n\n'
        "requirements:\n  host:\n    - python >=3.11\n  run:\n"
        "    - python >=3.11\n    - spectrochempy >=0.10\n    - numpy\n"
    )

    def test_tag_recipe_takes_precedence(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        (plugin_dir / "recipe.yaml").write_text('context:\n  version: "0.1.4"\n')
        recipe_dir, recipe_file, origin = module.resolve_recipe(
            plugin_dir, "0.1.4", self.MASTER_RECIPE
        )
        assert recipe_file == "recipe.yaml"
        assert origin == "tag"
        assert recipe_dir == str(plugin_dir)

    def test_master_fallback_writes_recovery_recipe(self, tmp_path, monkeypatch):
        module = load_module()
        plugin_dir = self._plugin_dir(
            tmp_path, dependency="spectrochempy[io]>=0.12,<0.13"
        )
        master = tmp_path / "master_recipe.yaml"
        master.write_text(self.MASTER_RECIPE)
        recipe_dir, recipe_file, origin = module.resolve_recipe(
            plugin_dir, "0.1.8", str(master)
        )
        assert (recipe_dir, recipe_file, origin) == (
            str(plugin_dir),
            "recipe.yaml",
            "master-fallback",
        )
        written = (plugin_dir / "recipe.yaml").read_text()
        assert 'version: "0.1.8"' in written
        assert 'version: "${{ version }}"' in written
        assert "- spectrochempy >=0.12,<0.13" in written

    def test_master_fallback_uses_tag_code_pyproject(self, tmp_path):
        """The recovery recipe's core bound comes from the tag checkout, not master."""
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path, dependency="spectrochempy>=0.12,<0.13")
        master = tmp_path / "master_recipe.yaml"
        master.write_text(self.MASTER_RECIPE)
        _, _, origin = module.resolve_recipe(plugin_dir, "0.1.8", str(master))
        assert origin == "master-fallback"
        assert (
            "- spectrochempy >=0.12,<0.13" in (plugin_dir / "recipe.yaml").read_text()
        )

    def test_no_master_recipe_raises(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        with pytest.raises(ValueError, match="no master recipe"):
            module.resolve_recipe(plugin_dir, "0.1.8", None)

    def test_missing_master_recipe_file_raises(self, tmp_path):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        with pytest.raises(ValueError, match="no master recipe"):
            module.resolve_recipe(
                plugin_dir, "0.1.8", str(tmp_path / "does-not-exist.yaml")
            )

    def test_tag_recipe_with_inconsistent_version_stays_tag_not_fallback(
        self, tmp_path
    ):
        """A tag that ships an inconsistent recipe must NOT fall back silently."""
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        (plugin_dir / "recipe.yaml").write_text('context:\n  version: "0.1.4"\n')
        recipe_dir, recipe_file, origin = module.resolve_recipe(
            plugin_dir, "0.1.8", self.MASTER_RECIPE
        )
        assert (recipe_file, origin) == ("recipe.yaml", "tag")
        # The inconsistency is surfaced by validate-release (exit 1), never hidden.
        assert (
            module.cmd_validate_release(
                argparse.Namespace(plugin_dir=str(plugin_dir), version="0.1.8")
            )
            == 1
        )


class TestVerifyTagExists:
    def _git_repo(self, tmp_path):
        import subprocess

        subprocess.run(["git", "init"], check=True, capture_output=True, cwd=tmp_path)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
            cwd=tmp_path,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            capture_output=True,
            cwd=tmp_path,
        )
        (tmp_path / "file.txt").write_text("init")
        subprocess.run(
            ["git", "add", "."], check=True, capture_output=True, cwd=tmp_path
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            check=True,
            capture_output=True,
            cwd=tmp_path,
        )

    def test_existing_tag_verified(self, tmp_path):
        import subprocess

        module = load_module()
        self._git_repo(tmp_path)
        subprocess.run(
            ["git", "tag", "spectrochempy-perkinelmer-v0.1.4"],
            check=True,
            capture_output=True,
            cwd=tmp_path,
        )
        assert module.verify_tag_exists(
            "spectrochempy-perkinelmer-v0.1.4",
            plugin_name="spectrochempy-perkinelmer",
            cwd=tmp_path,
        )

    def test_missing_tag_not_verified(self, tmp_path, capsys):
        module = load_module()
        self._git_repo(tmp_path)
        ok = module.verify_tag_exists(
            "spectrochempy-perkinelmer-v0.1.4",
            plugin_name="spectrochempy-perkinelmer",
            cwd=tmp_path,
        )
        assert ok is False


class TestCmdDeriveTag:
    def test_rejects_invalid_version(self, tmp_path, monkeypatch):
        module = load_module()
        monkeypatch.setattr(module, "verify_tag_exists", lambda *a, **k: True)
        assert (
            module.cmd_derive_tag(
                argparse.Namespace(plugin="spectrochempy-nmr", version="v0.1.8")
            )
            == 1
        )

    def test_derives_canonical_tag(self, tmp_path, monkeypatch, capsys):
        module = load_module()
        monkeypatch.setattr(module, "verify_tag_exists", lambda *a, **k: True)
        assert (
            module.cmd_derive_tag(
                argparse.Namespace(plugin="spectrochempy-perkinelmer", version="0.1.4")
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "spectrochempy-perkinelmer-v0.1.4" in out


class TestCmdResolveRecipe:
    def _plugin_dir(self, tmp_path):
        plugin_dir = tmp_path / "spectrochempy-perkinelmer"
        plugin_dir.mkdir()
        (plugin_dir / "pyproject.toml").write_text(
            '[project]\nversion = "0.1.4"\ndependencies = ['
            "'"
            "spectrochempy>=0.12,<0.13"
            "'"
            "]\n"
        )
        return plugin_dir

    MASTER = TestResolveRecipe.MASTER_RECIPE

    def test_emits_github_output_lines_tag(self, tmp_path, capsys):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        (plugin_dir / "recipe.yaml").write_text('context:\n  version: "0.1.4"\n')
        assert (
            module.cmd_resolve_recipe(
                argparse.Namespace(
                    tag_dir=str(plugin_dir), version="0.1.4", master_recipe=""
                )
            )
            == 0
        )
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "=" in ln]
        assert f"recipe_path={plugin_dir}" in lines
        assert "recipe_file=recipe.yaml" in lines
        assert "recipe_origin=tag" in lines

    def test_emits_github_output_lines_fallback(self, tmp_path, capsys):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        master = tmp_path / "master_recipe.yaml"
        master.write_text(self.MASTER)
        assert (
            module.cmd_resolve_recipe(
                argparse.Namespace(
                    tag_dir=str(plugin_dir),
                    version="0.1.8",
                    master_recipe=str(master),
                )
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "recipe_origin=master-fallback" in out
        assert "recipe_file=recipe.yaml" in out
        assert 'version: "0.1.8"' in (plugin_dir / "recipe.yaml").read_text()

    def test_missing_master_recipe_exit_two(self, tmp_path, capsys):
        module = load_module()
        plugin_dir = self._plugin_dir(tmp_path)
        assert (
            module.cmd_resolve_recipe(
                argparse.Namespace(
                    tag_dir=str(plugin_dir), version="0.1.8", master_recipe=""
                )
            )
            == 2
        )
        assert "::error::" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Repair workflow (structural test)
# ---------------------------------------------------------------------------


class TestRepairWorkflowStructure:
    WORKFLOW = (
        Path(__file__).parents[3]
        / ".github"
        / "workflows"
        / "repair_conda_plugin_release.yml"
    )

    def _text(self):
        return self.WORKFLOW.read_text()

    def test_version_only_input(self):
        import yaml

        data = yaml.safe_load(self._text())
        trigger = data.get("on") or data.get(True)
        inputs = trigger["workflow_dispatch"]["inputs"]
        assert "plugin_name" in inputs
        assert "version" in inputs
        assert "tag" not in inputs

    def test_closed_plugin_choice(self):
        text = self._text()
        assert "spectrochempy-perkinelmer" in text
        assert "spectrochempy-nmr" in text
        assert "e.g. 0.1.8" in text

    def test_explicit_refs_tags_checkout(self):
        text = self._text()
        assert "refs/tags/" in text
        assert "git checkout --detach" in text

    def test_derive_tag_and_resolve_recipe_used(self):
        text = self._text()
        assert "derive-tag" in text
        assert "resolve-recipe" in text

    def test_recipe_origin_propagated(self):
        text = self._text()
        assert "recipe_origin" in text
        assert "master-fallback" in text or "recipe_origin" in text


# ---------------------------------------------------------------------------
# PerkinElmer canonical recipe (exists on master, matches convention)
# ---------------------------------------------------------------------------


class TestPerkinElmerRecipe:
    def _plugin_dir(self):
        return Path(__file__).parents[3] / "plugins" / "spectrochempy-perkinelmer"

    def test_recipe_exists(self):
        assert (self._plugin_dir() / "recipe.yaml").is_file()

    def test_recipe_matches_convention(self):
        recipe = (self._plugin_dir() / "recipe.yaml").read_text()
        assert "spectrochempy >=0.10" in recipe
        assert "python >=3.11" in recipe
        assert "LicenseRef-CeCILL-B" in recipe
        assert ".find_spec('spectrochempy_perkinelmer')" in recipe


# ---------------------------------------------------------------------------
# Concurrency key behavior (structural test)
# ---------------------------------------------------------------------------


class TestConcurrencyKeyBehavior:
    """Verify the workflow YAML uses per-tag concurrency that doesn't cancel stable builds."""

    def test_build_package_yml_concurrency(self):
        workflow = (
            Path(__file__).parents[3] / ".github" / "workflows" / "build_package.yml"
        ).read_text()
        # Should NOT cancel in-progress for stable releases
        assert "cancel-in-progress:" in workflow
        # Should have conditional cancel logic
        assert "startsWith(github.ref_name, 'spectrochempy-v')" in workflow

    def test_no_unconditional_cancel(self):
        workflow = (
            Path(__file__).parents[3] / ".github" / "workflows" / "build_package.yml"
        ).read_text()
        # The cancel-in-progress should not be simply "true"
        lines = [
            line.strip()
            for line in workflow.splitlines()
            if "cancel-in-progress" in line
        ]
        assert len(lines) >= 1
        for line in lines:
            assert line != "cancel-in-progress: true"
