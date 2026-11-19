"""Regression coverage for generated Conda and pip dependency files."""

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "src/spectrochempy/ci/regenerate_requirements.py"


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def generator(tmp_path, monkeypatch, request):
    write_text = Path.write_text

    def write_with_newline(path, data, *args, **kwargs):
        kwargs["newline"] = request.param
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_with_newline)
    spec = importlib.util.spec_from_file_location("regenerate_requirements", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "repo_path", tmp_path)
    (tmp_path / "environments").mkdir()
    (tmp_path / "requirements").mkdir()
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    return module


def _dependencies(generator, suffix=""):
    path = generator.repo_path / "environments" / f"environment{suffix}.yml"
    return yaml.safe_load(path.read_text())["dependencies"]


@pytest.mark.parametrize("suffix", ["", "_test", "_dev", "_docs", "_build"])
def test_generated_environment_has_pip(generator, suffix):
    generator.main()
    dependencies = _dependencies(generator, suffix)
    assert dependencies.count("pip") == 1
    assert None not in dependencies


@pytest.mark.parametrize(
    "suffix, expected",
    [
        ("", []),
        ("_test", ["pytest-ruff"]),
        ("_docs", ["sphinx-tabs == 3.4.7"]),
        ("_dev", ["pytest-ruff", "sphinx-tabs == 3.4.7"]),
        ("_build", []),
    ],
)
def test_unavailable_conda_requirements_are_installed_with_pip(
    generator, suffix, expected
):
    generator.main()
    dependencies = _dependencies(generator, suffix)
    conda = [dep for dep in dependencies if isinstance(dep, str)]
    pip_sections = [dep["pip"] for dep in dependencies if isinstance(dep, dict)]
    assert pip_sections == ([expected] if expected else [])
    assert not any(dep.startswith(("pytest-ruff", "sphinx-tabs")) for dep in conda)
    if suffix in ("_docs", "_dev"):
        assert "sphinx=7.4.7" in conda
    assert "matplotlib-base" in conda or suffix == "_build"


def test_pip_requirements_are_preserved(generator):
    generator.main()
    for path in (generator.repo_path / "requirements").glob("*.txt"):
        assert path.read_text(encoding="utf-8") == (
            ROOT / "requirements" / path.name
        ).read_text(encoding="utf-8")


def test_regeneration_is_reproducible_and_generated_files_are_current(generator):
    generator.main()
    paths = list((generator.repo_path / "environments").glob("*.yml"))
    first = {path.name: path.read_bytes() for path in paths}
    generator.main()
    assert {path.name: path.read_bytes() for path in paths} == first
    for path in paths:
        assert path.read_text(encoding="utf-8") == (
            ROOT / "environments" / path.name
        ).read_text(encoding="utf-8")
        assert (
            _dependencies(generator, path.stem.removeprefix("environment")).count("pip")
            == 1
        )


def test_pip_only_core_and_optional_requirements_share_one_section(generator):
    requirement = "PyTest_Ruff[extra]>=0.4; python_version >= '3.11'"
    generator.generate_conda_environments(
        [requirement],
        {
            "test": [requirement, "numpy"],
            "dev": ["spectrochempy[test]", requirement],
            "build": ["setuptools"],
        },
        pip_only=["pytest.ruff"],
    )
    assert _dependencies(generator) == ["pip", {"pip": [requirement]}]
    for suffix in ("_test", "_dev"):
        dependencies = _dependencies(generator, suffix)
        assert [dep for dep in dependencies if isinstance(dep, dict)] == [
            {"pip": [requirement]}
        ]
        assert "numpy" in dependencies
        assert None not in dependencies
    assert _dependencies(generator, "_build") == ["pip", "setuptools"]
