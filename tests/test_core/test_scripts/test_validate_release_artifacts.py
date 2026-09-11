"""Tests for .github/workflows/scripts/validate_release_artifacts.py."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".github"
    / "workflows"
    / "scripts"
    / "validate_release_artifacts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_release_artifacts", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers: build synthetic wheel and sdist
# ---------------------------------------------------------------------------


def _make_wheel(
    tmp_path: Path,
    name: str = "testpkg",
    version: str = "1.0.0",
    build: str = "0",
    module_name: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Build a minimal valid wheel in tmp_path."""
    if module_name is None:
        module_name = name.replace("-", "_")
    dist_info = f"{module_name}-{version}.dist-info"
    wheel_name = f"{name}-{version}-{build}-py3-none-any.whl"
    wheel_path = tmp_path / wheel_name

    metadata = (
        f"Metadata-Version: 2.1\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        f"Requires-Python: >=3.11\n"
        f"License: MIT\n"
        f"Requires-Dist: numpy>=1.20\n"
        f"Requires-Dist: scipy\n"
    )
    wheel_content = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = (
        f"{module_name}/__init__.py,sha256=abc123,100\n"
        f"{module_name}/core.py,sha256=def456,200\n"
        f"{dist_info}/METADATA,sha256=ghi789,300\n"
        f"{dist_info}/WHEEL,sha256=jkl012,100\n"
        f"{dist_info}/RECORD,,\n"
    )

    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(f"{module_name}/__init__.py", f'__version__ = "{version}"\n')
        zf.writestr(f"{module_name}/core.py", "def main(): pass\n")
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr(f"{dist_info}/WHEEL", wheel_content)
        zf.writestr(f"{dist_info}/RECORD", record)
        if extra_files:
            for path, content in extra_files.items():
                zf.writestr(path, content)

    return wheel_path


def _make_sdist(
    tmp_path: Path,
    name: str = "testpkg",
    version: str = "1.0.0",
    module_name: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Build a minimal valid sdist in tmp_path."""
    if module_name is None:
        module_name = name.replace("-", "_")
    sdist_name = f"{name}-{version}.tar.gz"
    sdist_path = tmp_path / sdist_name
    pkg_info = (
        f"Metadata-Version: 2.1\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        f"Requires-Python: >=3.11\n"
        f"License: MIT\n"
        f"Requires-Dist: numpy>=1.20\n"
    )

    with tarfile.open(sdist_path, "w:gz") as tf:
        for fname, content in [
            (f"{name}-{version}/PKG-INFO", pkg_info),
            (f"{name}-{version}/setup.py", "from setuptools import setup; setup()"),
            (
                f"{name}-{version}/{module_name}/__init__.py",
                f'__version__ = "{version}"\n',
            ),
            (f"{name}-{version}/{module_name}/core.py", "def main(): pass\n"),
        ]:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=fname)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if extra_files:
            for fname, content in extra_files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=f"{name}-{version}/{fname}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    return sdist_path


def _make_conda(
    tmp_path: Path,
    name: str = "testpkg",
    version: str = "1.0.0",
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Build a minimal valid .conda file in tmp_path."""
    artifact_name = f"{name}-{version}-0_0.tar.bz2"
    artifact_path = tmp_path / artifact_name
    index_json = {
        "name": name,
        "version": version,
        "build": "0_0",
        "build_number": 0,
        "arch": "noarch",
        "subdir": "noarch",
        "depends": ["python >=3.11", "numpy"],
    }

    with tarfile.open(artifact_path, "w:bz2") as tf:
        data = json.dumps(index_json).encode("utf-8")
        info = tarfile.TarInfo(name="info/index.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        py_content = b"__version__ = '1.0.0'\n"
        py_info = tarfile.TarInfo(name=f"{name.replace('-', '_')}/__init__.py")
        py_info.size = len(py_content)
        tf.addfile(py_info, io.BytesIO(py_content))
        if extra_files:
            for fname, content in extra_files.items():
                info2 = tarfile.TarInfo(name=f"info/{fname}")
                info2.size = len(content)
                tf.addfile(info2, io.BytesIO(content))

    return artifact_path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverPythonArtifacts:
    def test_finds_wheel_and_sdist(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "1.0.0")
        _make_sdist(dist, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        result = module.discover_python_artifacts(dist, "testpkg", "1.0.0", report)
        assert result.wheel is not None
        assert result.sdist is not None
        assert report.passed

    def test_missing_dist_dir(self, tmp_path):
        module = load_module()
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        result = module.discover_python_artifacts(
            tmp_path / "nonexistent", "testpkg", "1.0.0", report
        )
        assert result.wheel is None
        assert not report.passed
        assert report.failure_count == 1

    def test_no_matching_wheel(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "otherpkg", "1.0.0")
        _make_sdist(dist, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        result = module.discover_python_artifacts(dist, "testpkg", "1.0.0", report)
        assert result.wheel is None
        assert not report.passed

    def test_multiple_wheels_warns(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "1.0.0", build="0")
        _make_wheel(dist, "testpkg", "1.0.0", build="1")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        result = module.discover_python_artifacts(dist, "testpkg", "1.0.0", report)
        assert result.wheel is not None
        assert result.extra_wheels
        assert report.warning_count >= 1

    def test_version_mismatch_warns(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "2.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        result = module.discover_python_artifacts(dist, "testpkg", "1.0.0", report)
        assert result.wheel is None

    def test_underscore_name_matches(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "test_pkg", "1.0.0")
        _make_sdist(dist, "test_pkg", "1.0.0")
        report = module.ValidationReport("test-pkg", "1.0.0", "python")
        result = module.discover_python_artifacts(dist, "test-pkg", "1.0.0", report)
        assert result.wheel is not None
        assert result.sdist is not None


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


class TestNormaliseDistName:
    def test_normalises_hyphens(self):
        module = load_module()
        assert module._normalise_dist_name("test-pkg") == "test-pkg"

    def test_normalises_underscores(self):
        module = load_module()
        assert module._normalise_dist_name("test_pkg") == "test-pkg"

    def test_normalises_dots(self):
        module = load_module()
        assert module._normalise_dist_name("test.pkg") == "test-pkg"

    def test_case_insensitive(self):
        module = load_module()
        assert module._normalise_dist_name("TestPkg") == "testpkg"


# ---------------------------------------------------------------------------
# Metadata reading
# ---------------------------------------------------------------------------


class TestReadWheelMetadata:
    def test_reads_metadata(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")
        text = module._read_wheel_metadata_text(wheel)
        assert "Name: testpkg" in text
        assert "Version: 1.0.0" in text

    def test_corrupt_wheel(self, tmp_path):
        module = load_module()
        corrupt = tmp_path / "corrupt.whl"
        corrupt.write_bytes(b"not a zip")
        text = module._read_wheel_metadata_text(corrupt)
        assert text == ""


class TestReadSdistMetadata:
    def test_reads_pkg_info(self, tmp_path):
        module = load_module()
        sdist = _make_sdist(tmp_path, "testpkg", "1.0.0")
        text = module._read_sdist_metadata_text(sdist)
        assert "Name: testpkg" in text
        assert "Version: 1.0.0" in text


class TestParseRfc822:
    def test_parses_simple_headers(self):
        module = load_module()
        text = "Name: test\nVersion: 1.0\nLicense: MIT\n"
        result = module._parse_rfc822(text)
        assert result == {"Name": "test", "Version": "1.0", "License": "MIT"}

    def test_ignores_continuation_lines(self):
        module = load_module()
        text = "Name: test\n Description: a long\n description\n"
        result = module._parse_rfc822(text)
        assert "Name" in result


class TestParseRequiresDist:
    def test_extracts_deps(self):
        module = load_module()
        text = (
            "Metadata-Version: 2.1\n"
            "Name: test\n"
            "Requires-Dist: numpy>=1.20\n"
            "Requires-Dist: scipy\n"
        )
        result = module._parse_requires_dist(text)
        assert result == ["numpy>=1.20", "scipy"]


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


class TestValidatePythonMetadata:
    def test_consistent_metadata(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")
        sdist = _make_sdist(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel, sdist=sdist)
        module.validate_python_metadata(artifacts, "testpkg", "1.0.0", report)
        assert report.passed
        assert any(
            c.name == "metadata:consistency" and c.severity == module.Severity.SUCCESS
            for c in report.checks
        )

    def test_name_mismatch(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "wrongname", "1.0.0")
        sdist = _make_sdist(tmp_path, "wrongname", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel, sdist=sdist)
        module.validate_python_metadata(artifacts, "testpkg", "1.0.0", report)
        assert not report.passed

    def test_version_mismatch(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "2.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_metadata(artifacts, "testpkg", "1.0.0", report)
        assert not report.passed

    def test_missing_metadata(self, tmp_path):
        module = load_module()
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        # Create a wheel without METADATA
        wheel_path = tmp_path / "testpkg-1.0.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel_path, "w") as zf:
            zf.writestr("testpkg/__init__.py", 'version = "1.0.0"\n')
        artifacts = module.PythonArtifacts(wheel=wheel_path)
        module.validate_python_metadata(artifacts, "testpkg", "1.0.0", report)
        has_failure = any(c.severity == module.Severity.FAILURE for c in report.checks)
        assert has_failure

    def test_inconsistent_wheel_sdist(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")
        sdist = _make_sdist(tmp_path, "testpkg", "2.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel, sdist=sdist)
        module.validate_python_metadata(artifacts, "testpkg", "1.0.0", report)
        assert not report.passed


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------


class TestValidatePythonContent:
    def test_clean_wheel(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_content(artifacts, "testpkg", report)
        assert report.passed

    def test_path_traversal_detected(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(
            tmp_path,
            "testpkg",
            "1.0.0",
            extra_files={"../../etc/passwd": "bad"},
        )
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_content(artifacts, "testpkg", report)
        assert not report.passed

    def test_absolute_path_detected(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(
            tmp_path,
            "testpkg",
            "1.0.0",
            extra_files={"/etc/passwd": "bad"},
        )
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_content(artifacts, "testpkg", report)
        assert not report.passed

    def test_sensitive_file_warning(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(
            tmp_path,
            "testpkg",
            "1.0.0",
            extra_files={"testpkg/.env": "SECRET=123"},
        )
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_content(artifacts, "testpkg", report)
        has_warning = any(
            c.severity == module.Severity.WARNING and "sensitive" in c.message.lower()
            for c in report.checks
        )
        assert has_warning

    def test_corrupt_wheel(self, tmp_path):
        module = load_module()
        corrupt = tmp_path / "corrupt.whl"
        corrupt.write_bytes(b"not a zip")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=corrupt)
        module.validate_python_content(artifacts, "testpkg", report)
        assert not report.passed

    def test_package_found_in_wheel(self, tmp_path):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.validate_python_content(artifacts, "testpkg", report)
        assert any(
            c.name == "content:package" and c.severity == module.Severity.SUCCESS
            for c in report.checks
        )


# ---------------------------------------------------------------------------
# Conda validation
# ---------------------------------------------------------------------------


class TestDiscoverCondaArtifact:
    def test_finds_valid_artifact(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        result = module.discover_conda_artifact(artifact, "testpkg", "1.0.0", report)
        assert result is not None
        assert report.passed

    def test_missing_file(self, tmp_path):
        module = load_module()
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        result = module.discover_conda_artifact(
            tmp_path / "nonexistent.conda", "testpkg", "1.0.0", report
        )
        assert result is None
        assert not report.passed

    def test_bad_extension(self, tmp_path):
        module = load_module()
        bad = tmp_path / "testpkg-1.0.0-0_0.zip"
        bad.write_bytes(b"data")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        result = module.discover_conda_artifact(bad, "testpkg", "1.0.0", report)
        assert result is None
        assert not report.passed

    def test_version_mismatch(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "testpkg", "2.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        result = module.discover_conda_artifact(artifact, "testpkg", "1.0.0", report)
        assert result is None
        assert not report.passed


class TestValidateCondaMetadata:
    def test_reads_metadata(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        info = module.validate_conda_metadata(artifact, "testpkg", "1.0.0", report)
        assert info["name"] == "testpkg"
        assert info["version"] == "1.0.0"
        assert report.passed

    def test_name_mismatch(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "wrongpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        module.validate_conda_metadata(artifact, "testpkg", "1.0.0", report)
        assert not report.passed


class TestValidateCondaContent:
    def test_clean_package(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "testpkg", "1.0.0")
        report = module.ValidationReport("testpkg", "1.0.0", "conda")
        module.validate_conda_content(artifact, "testpkg", report)
        assert report.passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_python_subcommand(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "1.0.0")
        _make_sdist(dist, "testpkg", "1.0.0")
        rc = module.main(
            [
                "python",
                "--package",
                "testpkg",
                "--version",
                "1.0.0",
                "--dist",
                str(dist),
            ]
        )
        assert rc == 0

    def test_python_missing_wheel_fails(self, tmp_path):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        rc = module.main(
            [
                "python",
                "--package",
                "testpkg",
                "--version",
                "1.0.0",
                "--dist",
                str(dist),
            ]
        )
        assert rc == 1

    def test_conda_subcommand(self, tmp_path):
        module = load_module()
        artifact = _make_conda(tmp_path, "testpkg", "1.0.0")
        rc = module.main(
            [
                "conda",
                "--package",
                "testpkg",
                "--version",
                "1.0.0",
                "--artifact",
                str(artifact),
            ]
        )
        assert rc == 0

    def test_json_output(self, tmp_path, capsys):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "1.0.0")
        _make_sdist(dist, "testpkg", "1.0.0")
        module.main(
            [
                "python",
                "--package",
                "testpkg",
                "--version",
                "1.0.0",
                "--dist",
                str(dist),
                "--json",
            ]
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["package"] == "testpkg"
        assert data["passed"] is True

    def test_markdown_output(self, tmp_path, capsys):
        module = load_module()
        dist = tmp_path / "dist"
        dist.mkdir()
        _make_wheel(dist, "testpkg", "1.0.0")
        _make_sdist(dist, "testpkg", "1.0.0")
        module.main(
            [
                "python",
                "--package",
                "testpkg",
                "--version",
                "1.0.0",
                "--dist",
                str(dist),
                "--markdown",
            ]
        )
        captured = capsys.readouterr()
        assert "## Artifact validation" in captured.out
        assert "PASSED" in captured.out


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_passed_when_no_failures(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("check1", module.Severity.SUCCESS, "ok")
        assert report.passed

    def test_failed_when_any_failure(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("check1", module.Severity.SUCCESS, "ok")
        report.add("check2", module.Severity.FAILURE, "bad")
        assert not report.passed

    def test_warnings_do_not_fail(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("check1", module.Severity.WARNING, "warn")
        assert report.passed

    def test_to_json(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("check1", module.Severity.SUCCESS, "ok")
        data = report.to_json()
        assert data["package"] == "pkg"
        assert len(data["checks"]) == 1

    def test_to_markdown(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("check1", module.Severity.SUCCESS, "ok")
        md = report.to_markdown()
        assert "## Artifact validation" in md
        assert "PASSED" in md

    def test_failure_count(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        report.add("c1", module.Severity.FAILURE, "f1")
        report.add("c2", module.Severity.FAILURE, "f2")
        report.add("c3", module.Severity.SUCCESS, "s1")
        assert report.failure_count == 2


# ---------------------------------------------------------------------------
# Archive path safety
# ---------------------------------------------------------------------------


class TestCheckArchivePaths:
    def test_traversal_detected(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        module._check_archive_paths(["../../etc/passwd"], report)
        assert report.failure_count >= 1

    def test_absolute_path_detected(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        module._check_archive_paths(["/etc/passwd"], report)
        assert report.failure_count >= 1

    def test_safe_paths_ok(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        module._check_archive_paths(["pkg/__init__.py", "pkg/core.py"], report)
        assert report.failure_count == 0

    def test_sensitive_file_warns(self):
        module = load_module()
        report = module.ValidationReport("pkg", "1.0", "python")
        module._check_archive_paths(["pkg/.env"], report)
        assert report.warning_count >= 1


# ---------------------------------------------------------------------------
# No shell=True and no upload protection
# ---------------------------------------------------------------------------


class TestNoShellOrUpload:
    def test_no_shell_true(self):
        source = SCRIPT_PATH.read_text()
        assert "shell=True" not in source

    def test_no_upload_commands(self):
        source = SCRIPT_PATH.read_text()
        assert "twine upload" not in source
        assert "anaconda upload" not in source
        assert "anaconda-client" not in source

    def test_uses_subprocess_list(self):
        source = SCRIPT_PATH.read_text()
        assert "subprocess.run(" in source


# ---------------------------------------------------------------------------
# Install & smoke test
# ---------------------------------------------------------------------------


class TestInstallSmokeTest:
    def test_venv_uses_system_site_packages_with_no_deps(self, tmp_path, monkeypatch):
        """With --no-deps the venv must reuse the base environment packages."""
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")

        calls: list[list[str]] = []

        def fake_run(cmd, capture_output=True, text=True):  # noqa: ARG001
            calls.append(cmd)
            if cmd[-2:] == ["-m", "venv"] or "--system-site-packages" in cmd:
                # Pretend venv creation succeeds; a real venv dir is not needed
                # because the import step is never reached in this test.
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if "pip" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "success", "")
            return subprocess.CompletedProcess(cmd, 0, "__version__ = '1.0.0'\n", "")

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.install_and_smoketest(
            artifacts, "testpkg", "1.0.0", None, no_deps=True, report=report
        )

        venv_cmd = next(c for c in calls if "--system-site-packages" in c)
        assert "--system-site-packages" in venv_cmd

    def test_venv_isolated_without_no_deps(self, tmp_path, monkeypatch):
        """Without --no-deps the venv must stay isolated from the base env."""
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")

        calls: list[list[str]] = []

        def fake_run(cmd, capture_output=True, text=True):  # noqa: ARG001
            calls.append(list(cmd))
            if "-m" in cmd and "venv" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if "pip" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "success", "")
            return subprocess.CompletedProcess(cmd, 0, "1.0.0\n", "")

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.install_and_smoketest(
            artifacts, "testpkg", "1.0.0", None, no_deps=False, report=report
        )

        venv_cmd = next(c for c in calls if "-m" in c and "venv" in c)
        assert "--system-site-packages" not in venv_cmd

    def test_import_failure_reported(self, tmp_path, monkeypatch):
        module = load_module()
        wheel = _make_wheel(tmp_path, "testpkg", "1.0.0")

        def fake_run(cmd, capture_output=True, text=True):  # noqa: ARG001
            if "-m" in cmd and "venv" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if "pip" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "success", "")
            return subprocess.CompletedProcess(
                cmd, 1, "", "ModuleNotFoundError: No module named 'testpkg'"
            )

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        report = module.ValidationReport("testpkg", "1.0.0", "python")
        artifacts = module.PythonArtifacts(wheel=wheel)
        module.install_and_smoketest(
            artifacts, "testpkg", "1.0.0", None, no_deps=True, report=report
        )

        assert not report.passed
        assert any(
            c.name == "install:import" and c.severity == module.Severity.FAILURE
            for c in report.checks
        )


# ---------------------------------------------------------------------------
# Plugin configuration
# ---------------------------------------------------------------------------


class TestPluginConfig:
    def test_official_plugins_defined(self):
        module = load_module()
        assert "spectrochempy-nmr" in module.OFFICIAL_PLUGINS
        assert "spectrochempy-perkinelmer" in module.OFFICIAL_PLUGINS
        assert len(module.OFFICIAL_PLUGINS) == 6

    def test_get_module_name_core(self):
        module = load_module()
        assert module.get_module_name("spectrochempy") == "spectrochempy"

    def test_get_module_name_plugin(self):
        module = load_module()
        assert module.get_module_name("spectrochempy-nmr") == "spectrochempy_nmr"

    def test_get_module_name_unknown(self):
        module = load_module()
        assert module.get_module_name("unknown-package") is None
