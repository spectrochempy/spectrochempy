"""Test provenance and build selection for the Read the Docs entry point."""

import hashlib
import importlib
from contextlib import redirect_stderr
from io import BytesIO
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")
pytest.importorskip("jupytext")
pytest.importorskip("sphinx")

rtd_build = importlib.import_module("docs.rtd_build")


def test_case_distinct_api_names_have_distinct_gallery_labels(tmp_path, monkeypatch):
    pytest.importorskip("sphinx_gallery")
    from sphinx.application import Sphinx
    from sphinx.ext.autosummary.generate import AutosummaryRenderer

    import spectrochempy as scp

    monkeypatch.setattr(scp.Gaussian, "__doc__", "A Gaussian model.")
    monkeypatch.setattr(scp.gaussian, "__doc__", "A Gaussian profile.")
    sources = tmp_path / "sources"
    examples = tmp_path / "examples"
    sources.mkdir()
    examples.mkdir()
    (examples / "README.rst").write_text("Examples\n========\n")
    (examples / "plot_models.py").write_text(
        '"""Gaussian models\n===============\n"""\n'
        "import spectrochempy as scp\n"
        "model = scp.Gaussian()\n"
        "values = scp.gaussian([0., 1., 2.], pos=1., width=1.)\n"
    )
    templates = Path(__file__).resolve().parents[2] / "docs" / "_templates"
    (sources / "conf.py").write_text(
        'extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", '
        '"sphinx_gallery.gen_gallery"]\n'
        f"templates_path = [{str(templates)!r}]\n"
        "sphinx_gallery_conf = {\n"
        f"    'examples_dirs': {str(examples)!r},\n"
        "    'gallery_dirs': 'gallery',\n"
        "    'backreferences_dir': 'backreferences',\n"
        "    'doc_module': ('spectrochempy',),\n"
        "    'plot_gallery': False,\n"
        "    'download_all_examples': False,\n"
        "}\n"
    )
    (sources / "index.rst").write_text(
        "API\n===\n\n.. toctree::\n\n   model\n   function\n   gallery/index\n"
    )
    warnings = StringIO()
    app = Sphinx(
        str(sources),
        str(sources),
        str(tmp_path / "html"),
        str(tmp_path / "doctrees"),
        "html",
        status=StringIO(),
        warning=warnings,
        freshenv=True,
        warningiserror=True,
    )
    renderer = AutosummaryRenderer(app)
    for page, objname, template in (
        ("model", "Gaussian", "class"),
        ("function", "gaussian", "function"),
    ):
        content = renderer.render(
            template,
            {
                "fullname": f"spectrochempy.{objname}",
                "module": "spectrochempy",
                "objname": objname,
                "methods": [],
                "attributes": [],
            },
        )
        (sources / f"{page}.rst").write_text(content)
    app.build()

    assert app.statuscode == 0, warnings.getvalue()
    assert "duplicate label" not in warnings.getvalue()
    for page in ("model", "function"):
        assert "plot_models.html" in (tmp_path / "html" / f"{page}.html").read_text()


def test_gallery_local_class_backreferences_are_portable(tmp_path, monkeypatch):
    pytest.importorskip("sphinx_gallery")
    from sphinx.application import Sphinx

    ci = rtd_build.ci_artifacts
    sources = tmp_path / "docs" / "sources"
    examples = tmp_path / "examples"
    sources.mkdir(parents=True)
    examples.mkdir()
    (examples / "README.rst").write_text("Examples\n========\n")
    (examples / "plot_local_class.py").write_text(
        '''"""Local axes\n==========\n"""
import spectrochempy as scp

dataset = scp.NDDataset([1., 2., 3.])

def make_axes():
    class Axes:
        def plot(self):
            return None
    Axes.__module__ = "spectrochempy"
    return Axes()

axes = make_axes()
axes.plot()
'''
    )
    (sources / "conf.py").write_text(
        f"""extensions = ["sphinx_gallery.gen_gallery"]
master_doc = "index"
sphinx_gallery_conf = {{
    "examples_dirs": {str(examples)!r},
    "gallery_dirs": {ci.GALLERY.as_posix()!r},
    "backreferences_dir": {(ci.GALLERY / "backreferences").as_posix()!r},
    "doc_module": ("spectrochempy",),
    "download_all_examples": False,
}}
"""
    )
    (sources / "index.rst").write_text(
        f"Gallery\n=======\n\n.. toctree::\n\n   {ci.GALLERY.as_posix()}/index\n"
        "\n.. minigallery:: spectrochempy.NDDataset\n"
    )

    def generate(**_kwargs):
        app = Sphinx(
            str(sources),
            str(sources),
            str(tmp_path / "html"),
            str(tmp_path / "doctrees"),
            "html",
            status=StringIO(),
            warning=StringIO(),
            freshenv=True,
        )
        app.build()
        assert app.statuscode == 0

    monkeypatch.setattr(ci, "SOURCES", sources)
    monkeypatch.setattr(ci, "_produce_gallery", generate)
    monkeypatch.setattr(ci, "_commit", lambda: "commit")
    bundle = tmp_path / "bundle"
    ci.produce("gallery", bundle)
    root, files = ci._verified_files(bundle, "gallery", "commit")
    assert all(
        not set('<>:"\\|?*').intersection(path.relative_to(root).as_posix())
        for path in files
    )
    backreferences = (root / ci.BACKREFERENCES).read_text()
    assert "spectrochempy.make_axes.<locals>.Axes" in backreferences
    assert "spectrochempy.NDDataset" in backreferences
    assert any("_locals_" in path.name for path in files)
    assert "plot_local_class.html" in (tmp_path / "html" / "index.html").read_text()
    example = next(path for path in files if path.name == "plot_local_class.rst")
    example.write_text(example.read_text() + "altered")
    with pytest.raises(RuntimeError, match="Missing or altered files"):
        ci._verified_files(bundle, "gallery", "commit")


def test_selects_both_artifacts_from_one_successful_run(monkeypatch):
    commit = "a" * 40
    runs = [
        {"id": 1, "head_sha": "b" * 40, "event": "push"},
        {"id": 2, "head_sha": commit, "event": "push"},
        {"id": 3, "head_sha": commit, "event": "push"},
        {"id": 4, "head_sha": commit, "event": "workflow_dispatch"},
    ]
    for run in runs:
        run.update(status="completed", conclusion="success")

    def artifact(kind, artifact_id, expired=False):
        return {"name": f"docs-{kind}-{commit}", "id": artifact_id, "expired": expired}

    responses = {
        2: [artifact("tutorials", 20)],
        3: [artifact("gallery", 30), artifact("tutorials", 31, expired=True)],
        4: [artifact("gallery", 40), artifact("tutorials", 41)],
    }

    def github(path, token):
        assert token == "test-token"  # noqa: S105
        if path.startswith("workflows/"):
            assert f"head_sha={commit}" in path and "status=" not in path
            return {"workflow_runs": runs}
        return {"artifacts": responses[int(path.split("/")[1])]}

    monkeypatch.setattr(rtd_build, "_json", github)
    run_id, selected = rtd_build._find_artifacts(commit, "test-token", set())
    assert run_id == 4
    assert {kind: item["id"] for kind, item in selected.items()} == {
        "gallery": 40,
        "tutorials": 41,
    }


def test_zip_checksum_is_checked_before_extraction(tmp_path, monkeypatch):
    data = BytesIO()
    with ZipFile(data, "w") as bundle:
        bundle.writestr("manifest.json", "{}")
    content = data.getvalue()
    monkeypatch.setattr(rtd_build, "_request", lambda *_: BytesIO(content))
    destination = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        rtd_build._download({"id": 1, "digest": "wrong"}, "token", destination)
    assert not destination.exists()
    digest = hashlib.sha256(content).hexdigest()
    rtd_build._download({"id": 1, "digest": f"sha256:{digest}"}, "token", destination)
    assert (destination / "manifest.json").read_text() == "{}"


def test_artifact_mode_removes_token_from_environment(monkeypatch):
    monkeypatch.setenv("SCPY_RTD_GITHUB_TOKEN", "test-token")
    called = []

    def render(token):
        assert token == "test-token"  # noqa: S105
        assert rtd_build.os.environ.get("SCPY_RTD_GITHUB_TOKEN") is None
        called.append("artifact")

    monkeypatch.setattr(rtd_build, "_render_from_actions", render)
    rtd_build.main()
    assert called == ["artifact"]


@pytest.mark.parametrize("expired", [True, False], ids=["expired", "version-mismatch"])
def test_unusable_artifact_retries_the_entire_pair(monkeypatch, expired):
    monkeypatch.setattr(rtd_build.ci_artifacts, "_commit", lambda: "commit")
    rendered = []

    def find(_commit, _token, unavailable):
        if unavailable:
            assert unavailable == ({2} if expired else {1, 2})
            return 20, {"tutorials": {"id": 3}, "gallery": {"id": 4}}
        return 10, {"tutorials": {"id": 1}, "gallery": {"id": 2}}

    def download(artifact, _token, path):
        if expired and artifact["id"] == 2:
            raise HTTPError("https://api.github.com", 410, "Gone", {}, None)
        path.write_text(str(artifact["id"]))

    def render(root):
        pair = tuple((root / kind).read_text() for kind in ("tutorials", "gallery"))
        rendered.append(pair)
        if pair == ("1", "2"):
            raise rtd_build.ci_artifacts.ArtifactVersionMismatchError("wrong version")

    monkeypatch.setattr(rtd_build, "_find_artifacts", find)
    monkeypatch.setattr(rtd_build, "_download", download)
    monkeypatch.setattr(rtd_build.ci_artifacts, "render", render)
    rtd_build._render_from_actions("test-token")
    assert rendered == ([("3", "4")] if expired else [("1", "2"), ("3", "4")])


def test_same_commit_with_different_installed_version_is_rejected(tmp_path):
    commit = "a" * 40
    (tmp_path / "manifest.json").write_text(
        f'{{"commit": "{commit}", "kind": "tutorials", "version": "0.dev0"}}'
    )
    with pytest.raises(rtd_build.ci_artifacts.ArtifactVersionMismatchError):
        rtd_build.ci_artifacts._verified_files(tmp_path, "tutorials", commit)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_failed_producer_stops_waiting_with_run_link(monkeypatch, conclusion):
    monkeypatch.setattr(
        rtd_build,
        "_json",
        lambda *_: {
            "workflow_runs": [
                {
                    "id": 42,
                    "head_sha": "commit",
                    "event": "push",
                    "status": "completed",
                    "conclusion": conclusion,
                }
            ]
        },
    )
    with pytest.raises(RuntimeError, match=r"actions/runs/42") as error:
        rtd_build._find_artifacts("commit", "token", set())
    assert conclusion in str(error.value)


def test_active_retry_keeps_waiting_after_failed_run(monkeypatch):
    monkeypatch.setattr(
        rtd_build,
        "_json",
        lambda *_: {
            "workflow_runs": [
                {
                    "id": 43,
                    "head_sha": "commit",
                    "event": "workflow_dispatch",
                    "status": "in_progress",
                    "conclusion": None,
                },
                {
                    "id": 42,
                    "head_sha": "commit",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "failure",
                },
                {
                    "id": 44,
                    "head_sha": "other",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "failure",
                },
            ]
        },
    )
    assert rtd_build._find_artifacts("commit", "token", set()) is None


def test_older_complete_pair_is_used_despite_newer_failure(monkeypatch):
    def github(path, _token):
        if path.startswith("workflows/"):
            return {
                "workflow_runs": [
                    {
                        "id": 43,
                        "head_sha": "commit",
                        "event": "push",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    {
                        "id": 42,
                        "head_sha": "commit",
                        "event": "push",
                        "status": "completed",
                        "conclusion": "success",
                    },
                ]
            }
        return {
            "artifacts": [
                {"id": i, "name": f"docs-{kind}-commit", "expired": False}
                for i, kind in enumerate(("gallery", "tutorials"))
            ]
        }

    monkeypatch.setattr(rtd_build, "_json", github)
    assert rtd_build._find_artifacts("commit", "token", set())[0] == 42


def test_gallery_failure_traceback_escapes_captured_stderr(
    monkeypatch, tmp_path, capfd
):
    ci = rtd_build.ci_artifacts
    monkeypatch.setattr(ci, "_files", lambda *_: [])

    def fail(**_kwargs):
        raise RuntimeError("gallery diagnostic sentinel")

    monkeypatch.setattr(ci, "_produce_gallery", fail)
    captured = StringIO()
    with (
        redirect_stderr(captured),
        pytest.raises(RuntimeError, match="gallery diagnostic sentinel"),
    ):
        ci.produce("gallery", tmp_path / "bundle")
    stderr = capfd.readouterr().err
    assert "Traceback (most recent call last)" in stderr
    assert "RuntimeError: gallery diagnostic sentinel" in stderr
    assert "FAILED execute gallery" in stderr
    assert captured.getvalue() == ""


def test_watchdog_is_cancelled_after_failure(monkeypatch):
    from docs import build_diagnostics as diagnostics

    calls = []
    monkeypatch.setattr(
        diagnostics.faulthandler,
        "dump_traceback_later",
        lambda delay, **kwargs: calls.append((delay, kwargs)),
    )
    monkeypatch.setattr(
        diagnostics.faulthandler,
        "cancel_dump_traceback_later",
        lambda: calls.append("cancel"),
    )
    with pytest.raises(RuntimeError), diagnostics._diagnostics():
        raise RuntimeError("test")
    assert calls == [
        (120, {"repeat": True, "file": diagnostics.sys.__stderr__}),
        "cancel",
    ]


def test_public_mode_does_not_require_or_retain_github_credentials(monkeypatch):
    monkeypatch.setenv("SCPY_RTD_GITHUB_RESULTS", "1")
    monkeypatch.setenv("READTHEDOCS_VERSION", "latest")
    monkeypatch.setenv("SCPY_RTD_GITHUB_TOKEN", "test-token")
    called = []

    def render():
        assert "SCPY_RTD_GITHUB_TOKEN" not in rtd_build.os.environ
        called.append(True)

    monkeypatch.setattr(rtd_build, "_render_public_results", render)
    rtd_build.main()
    assert called == [True]
