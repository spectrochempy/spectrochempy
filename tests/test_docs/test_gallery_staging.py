# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for Sphinx-Gallery example staging."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = REPO_ROOT / "docs" / "conf.py"


def _load_docs_conf(monkeypatch):
    monkeypatch.setenv("SPHINX_PATTERN", "index")
    module_name = "_spectrochempy_docs_conf_for_tests"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_external_plugin_manifest(root: Path) -> Path:
    examples = root / "examples"
    source = examples / "core" / "c_importer" / "plot_external_reader.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        textwrap.dedent(
            '''
            """
            External reader example
            =======================
            """

            # sphinx_gallery_thumbnail_number = 1

            # %%
            import spectrochempy as scp

            dataset = scp.read("irdata/nh4y-activation.spg")
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (source.parent / "readme.rst").write_text(
        ".. _examples-importer-index:\n\nImport / Export\n---------------\n",
        encoding="utf-8",
    )
    manifest = examples / "gallery.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [plugin]
            name = "spectrochempy-external"
            title = "External plugin"

            [[examples]]
            path = "core/c_importer/plot_external_reader.py"
            section = "Import / Export"
            section_ref = "examples-importer-index"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return manifest


def test_plugin_manifest_examples_are_staged_once(monkeypatch, tmp_path):
    conf = _load_docs_conf(monkeypatch)
    manifest = _write_external_plugin_manifest(tmp_path / "external-plugin")

    project = tmp_path / "project"
    (project / "plugins").mkdir(parents=True)
    monkeypatch.setattr(conf, "PROJECT", project)
    monkeypatch.setattr(conf, "BUILDIR", tmp_path / "build")
    monkeypatch.setattr(conf, "example_source_dir", tmp_path / "empty-examples")
    monkeypatch.setattr(
        conf, "gallery_sections", ("core", "processing", "analysis", "plugins")
    )
    monkeypatch.setenv("SCP_PLUGIN_GALLERY_MANIFESTS", str(manifest))

    entries = conf._load_plugin_gallery_entries()
    staged = conf._stage_gallery_examples()
    conf._write_plugin_gallery_readmes(staged, entries)

    canonical = staged / "core" / "c_importer" / "plot_external_reader.py"
    plugin_copy = staged / "plugins" / "spectrochempy-external"

    assert canonical.exists()
    assert not plugin_copy.exists()
    assert not list((staged / "plugins").glob("**/plot_*.py"))
    assert not list((staged / "plugins").glob("**/*.ipynb"))


def test_plugin_index_references_canonical_gallery_page(monkeypatch, tmp_path):
    conf = _load_docs_conf(monkeypatch)
    manifest = _write_external_plugin_manifest(tmp_path / "external-plugin")

    project = tmp_path / "project"
    (project / "plugins").mkdir(parents=True)
    monkeypatch.setattr(conf, "PROJECT", project)
    monkeypatch.setenv("SCP_PLUGIN_GALLERY_MANIFESTS", str(manifest))

    entries = conf._load_plugin_gallery_entries()
    readme = conf._plugin_gallery_readme(entries)

    assert (
        ":ref:`sphx_glr_gettingstarted_examples_gallery_auto_examples_core_c_importer_plot_external_reader.py`"
        in readme
    )
    assert "``spectrochempy-external``" in readme
    assert ":ref:`Import / Export <examples-importer-index>`" in readme


def test_generated_credits_file_keeps_final_newline(monkeypatch):
    _load_docs_conf(monkeypatch)

    credits = REPO_ROOT / "docs" / "sources" / "credits" / "credits.rst"
    assert credits.read_bytes().endswith(b"\n")
