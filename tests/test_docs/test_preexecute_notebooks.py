"""Validate the separate notebook execution diagnostic without scientific data."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")
pytest.importorskip("sphinx")

runner = importlib.import_module("docs.preexecute_notebooks")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(runner, "SOURCES", sources)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    return sources, manifest


def _notebook(source="print('executed')", **metadata):
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source)], metadata=metadata
    )


def _write_manifest(sources, notebooks, **settings):
    defaults = {
        "execute": "auto",
        "kernel_name": "python3",
        "execute_arguments": [],
        "allow_errors": True,
        "timeout": 90,
    }
    defaults.update(settings)
    entries = {}
    for name, notebook in notebooks.items():
        path = sources / name
        nbformat.write(notebook, path)
        entries[name] = {
            "execute": runner._should_execute(notebook, defaults["execute"]),
            "input_sha256": runner._digest(path),
        }
    runner._save({"settings": defaults, "notebooks": entries})


def test_selection_preserves_nbsphinx_auto_and_metadata():
    notebook = _notebook()
    assert runner._should_execute(notebook, "auto")
    notebook.cells[0].execution_count = 1
    assert not runner._should_execute(notebook, "auto")
    notebook.metadata.nbsphinx = {"execute": "always"}
    assert runner._should_execute(notebook, "auto")
    notebook.metadata.nbsphinx.execute = "never"
    assert not runner._should_execute(notebook, "auto")
    assert not runner._should_execute(_notebook(""), "auto")
    notebook.metadata.nbsphinx.execute = "invalid"
    with pytest.raises(ValueError, match="Invalid"):
        runner._should_execute(notebook, "auto")


def test_prepare_uses_existing_preparation_and_configuration(workspace, monkeypatch):
    sources, manifest = workspace
    first = sources / "first.ipynb"
    excluded = sources / "excluded.ipynb"
    calls = []

    def prepare():
        calls.append("prepare")
        nbformat.write(_notebook(), first)
        nbformat.write(_notebook(), excluded)

    builder = SimpleNamespace(
        _prepare_build=prepare,
        _get_notebook_files=lambda: {first.with_suffix(""), excluded.with_suffix("")},
    )
    monkeypatch.setattr(runner, "BuildDocumentation", lambda: builder)
    settings = {
        "nbsphinx_execute": "auto",
        "nbsphinx_kernel_name": "python3",
        "nbsphinx_execute_arguments": ["--InlineBackend.rc={'figure.dpi': 96}"],
        "nbsphinx_allow_errors": True,
        "nbsphinx_timeout": 90,
        "sphinx_gallery_conf": {"plot_gallery": True},
        "exclude_patterns": ["excluded.ipynb"],
        "nbsphinx_exclude_patterns": [],
    }
    monkeypatch.setattr("sphinx.config.eval_config_file", lambda *_: settings)
    runner.prepare()
    result = json.loads(manifest.read_text())
    assert calls == ["prepare"]
    assert set(result["notebooks"]) == {"first.ipynb"}
    assert (
        result["settings"]["execute_arguments"]
        == settings["nbsphinx_execute_arguments"]
    )


def test_real_kernel_saves_outputs_cwd_and_allowed_errors(workspace):
    sources, manifest = workspace
    (sources / "input.txt").write_text("local input")
    notebook = _notebook(
        "import os\nfrom pathlib import Path\n"
        f"assert os.environ['PYTHONPATH'].split(os.pathsep)[0] == {str(runner.DOCS.parent / 'src')!r}\n"
        "print(Path('input.txt').read_text())"
    )
    notebook.cells.append(nbformat.v4.new_code_cell("raise ValueError('expected')"))
    _write_manifest(sources, {"sample.ipynb": notebook})
    runner.execute("sample.ipynb")
    saved = nbformat.read(sources / "sample.ipynb", as_version=4)
    assert saved.cells[0].outputs[0].text.strip() == "local input"
    assert saved.cells[1].outputs[0].ename == "ValueError"
    assert saved.metadata.nbsphinx.execute == "never"
    assert (
        json.loads(manifest.read_text())["notebooks"]["sample.ipynb"]["error_outputs"]
        == 1
    )
    runner.verify()
    with pytest.raises(RuntimeError, match="already processed"):
        runner.execute("sample.ipynb")


def test_real_kernels_isolate_preferences_between_notebooks(workspace, monkeypatch):
    sources, _ = workspace
    user_config = sources / "user-config"
    user_config.mkdir()
    user_preferences = user_config / "PlotPreferences.json"
    saved_preferences = json.dumps({"PlotPreferences": {"lines_linewidth": 7.5}})
    user_preferences.write_text(saved_preferences)
    monkeypatch.setenv("SCP_CONFIG_HOME", str(user_config))
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "import spectrochempy as scp\n"
        "from spectrochempy.application.application import app\n"
        "prefs = scp.preferences\n"
        "assert prefs.lines.linewidth == "
        "app.plot_preferences.traits()['lines_linewidth'].default_value\n"
        "prefs.lines.linewidth = 9.5\n"
        "config_home = Path(os.environ['SCP_CONFIG_HOME'])\n"
        "assert (config_home / 'PlotPreferences.json').is_file()\n"
        "print(config_home)"
    )
    _write_manifest(
        sources,
        {"first.ipynb": _notebook(source), "second.ipynb": _notebook(source)},
        allow_errors=False,
    )
    config_homes = []
    for name in ("first.ipynb", "second.ipynb"):
        runner.execute(name)
        saved = nbformat.read(sources / name, as_version=4)
        output = next(
            output
            for output in saved.cells[0].outputs
            if output.output_type == "stream"
        )
        config_home = Path(output.text.strip())
        assert config_home != user_config
        assert not config_home.exists()
        assert os.environ["SCP_CONFIG_HOME"] == str(user_config)
        assert user_preferences.read_text() == saved_preferences
        config_homes.append(config_home)
    assert config_homes[0] != config_homes[1]


@pytest.mark.parametrize("inherited_config", [False, True])
def test_failure_stops_and_cannot_pass_verification(
    workspace, monkeypatch, inherited_config
):
    sources, _ = workspace
    _write_manifest(sources, {"sample.ipynb": _notebook()})
    if inherited_config:
        monkeypatch.setenv("SCP_CONFIG_HOME", str(sources))
    else:
        monkeypatch.delenv("SCP_CONFIG_HOME", raising=False)
    config_homes = []

    class FailingProcessor:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 90
            assert kwargs["allow_errors"] is True

        def preprocess(self, *_):
            config_home = Path(os.environ["SCP_CONFIG_HOME"])
            assert config_home.is_dir()
            assert config_home != sources
            config_homes.append(config_home)
            raise TimeoutError("cell timeout")

    monkeypatch.setattr(runner, "ExecutePreprocessor", FailingProcessor)
    with pytest.raises(TimeoutError, match="cell timeout"):
        runner.execute("sample.ipynb")
    assert len(config_homes) == 1
    assert not config_homes[0].exists()
    assert os.environ.get("SCP_CONFIG_HOME") == (
        str(sources) if inherited_config else None
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.verify()


def test_metadata_overrides_and_skipped_notebooks(workspace, monkeypatch):
    sources, _ = workspace
    notebooks = {
        "sample.ipynb": _notebook(nbsphinx={"timeout": 3, "allow_errors": False}),
        "skip.ipynb": _notebook(nbsphinx={"execute": "never"}),
    }
    _write_manifest(sources, notebooks)
    calls = []

    class Processor:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def preprocess(self, notebook, resources):
            assert resources["metadata"]["path"] == str(sources)

    monkeypatch.setattr(runner, "ExecutePreprocessor", Processor)
    runner.execute("sample.ipynb")
    runner.execute("skip.ipynb")
    assert len(calls) == 1
    assert calls[0]["timeout"] == 3
    assert calls[0]["allow_errors"] is False
    runner.verify()
    with (sources / "sample.ipynb").open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="changed"):
        runner.verify()


def test_rtd_commands_cover_inventory_and_preserve_gallery():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / ".readthedocs.yaml").read_text())
    commands = config["build"]["jobs"]["build"]["html"]
    expected = sorted(
        path.with_suffix(".ipynb").relative_to(runner.SOURCES).as_posix()
        for path in runner.BuildDocumentation()._get_notebook_files()
    )
    assert len(expected) == 38
    assert commands[1:-2] == [
        f"python -u -m docs.preexecute_notebooks execute {name}" for name in expected
    ]
    assert commands[0].endswith("docs.preexecute_notebooks prepare")
    assert commands[-2].endswith("docs.preexecute_notebooks verify")
    assert commands[-1] == (
        "SCPY_DOCS_PREEXECUTED=1 SCPY_DOCS_TRACE=1 python -u docs/make.py "
        "html --warning-is-error --no-sync --no-data -j auto"
    )


def test_preexecution_cli_is_not_collected_as_a_notebook_example():
    examples = importlib.import_module("tests.test_docs.test_py_in_docs")
    assert Path(runner.__file__).resolve() not in examples.scripts
