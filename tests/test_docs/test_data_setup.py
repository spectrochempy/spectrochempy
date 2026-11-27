"""Check bundled documentation inputs and isolated notebook data selection."""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from docs import data_setup


def test_bundled_data_matches_manifest():
    data_setup.verify_data()
    manifest = json.loads((data_setup.DATA / "manifest.json").read_text())
    assert not any("corn" in name.lower() for name in manifest["files"])


@pytest.mark.parametrize("change", ["missing", "size", "hash"])
def test_invalid_data_fails_without_downloading(tmp_path, monkeypatch, change):
    payload = b"original"
    fixture = tmp_path / "input.dat"
    fixture.write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "input.dat": {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                }
            }
        )
    )
    monkeypatch.setattr(data_setup, "DATA", tmp_path)
    if change == "missing":
        fixture.unlink()
    else:
        fixture.write_bytes(b"modified" if change == "hash" else b"short")
    with pytest.raises(RuntimeError, match="documentation data"):
        data_setup.verify_data()


def test_notebook_kernel_uses_bundled_data(tmp_path, monkeypatch):
    pytest.importorskip("sphinx")
    nbformat = pytest.importorskip("nbformat")
    from nbconvert.preprocessors import ExecutePreprocessor
    from sphinx.config import eval_config_file

    for key in ("SCPY_CORN_DATA", "SCPY_CORN_ZIP", "SCPY_CORN_SYNTHETIC"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("SPHINX_PATTERN", "index")
    sources = tmp_path / "sources"
    shutil.copytree(
        data_setup.DATA.parent / "sources",
        sources,
        ignore=shutil.ignore_patterns("*.ipynb", "gallery", "generated"),
    )
    monkeypatch.setenv("SPHINX_SRCDIR", str(sources))
    monkeypatch.setenv("SCP_CONFIG_HOME", str(tmp_path / "user-config"))
    config = eval_config_file(str(data_setup.DATA.parent / "conf.py"), None)
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import spectrochempy as scp; "
                f"assert str(scp.preferences.datadir) == {str(data_setup.DATA)!r}; "
                "scp.preferences.reset(); "
                f"assert str(scp.preferences.datadir) == {str(data_setup.DATA)!r}",
            )
        ]
    )
    ExecutePreprocessor(
        kernel_name="python3",
        extra_arguments=config["nbsphinx_execute_arguments"],
        allow_errors=False,
        timeout=90,
    ).preprocess(notebook, {"metadata": {"path": str(tmp_path)}})


@pytest.mark.parametrize("singledoc", [None, "spectrochempy.NDDataset"])
def test_build_preparation_preserves_user_preferences(tmp_path, singledoc):
    config = tmp_path / "user-config"
    config.mkdir()
    preferences = config / "GeneralPreferences.json"
    saved = json.dumps({"GeneralPreferences": {"datadir": str(tmp_path / "own-data")}})
    preferences.write_text(saved)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from docs import make; "
            f"make.DOCS = Path({str(tmp_path / 'docs')!r}); "
            f"builder = make.BuildDocumentation(nosync=True, singledoc={singledoc!r}); "
            "builder._determine_version = lambda: ('1.0.0', None, 'latest'); "
            "builder._get_previous_versions = lambda: []; "
            "builder._prepare_build(); "
            "from spectrochempy import preferences; "
            f"assert str(preferences.datadir) == {str(data_setup.DATA)!r}",
        ],
        cwd=data_setup.DATA.parent.parent,
        env={
            **os.environ,
            "SCP_CONFIG_HOME": str(config),
            "SCPY_BUILDDIR": str(tmp_path / "build"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert preferences.read_text() == saved
    assert {path.name for path in config.iterdir()} == {"GeneralPreferences.json"}
