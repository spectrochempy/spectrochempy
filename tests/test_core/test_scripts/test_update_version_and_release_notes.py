"""
Tests for the RC-aware release notes machinery in
.github/workflows/scripts/update_version_and_release_notes.py.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[3] / ".github" / "workflows" / "scripts"
SCRIPT_PATH = SCRIPTS / "update_version_and_release_notes.py"

_module = None


def load_module():
    global _module
    if _module is not None:
        return _module
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "update_version_and_release_notes", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    _module = module
    return module


@pytest.fixture
def update_module():
    return load_module()


@pytest.fixture
def fake_whatsnew(update_module, tmp_path):
    update_module.WN = tmp_path
    for name in ("v0.12.8.rst", "v1.0.0rc1.rst", "v1.0.0.rst", "scripts-notes.rst"):
        (tmp_path / name).write_text("content", encoding="utf-8")
    return tmp_path


def test_release_index_includes_rc_notes(update_module, fake_whatsnew):
    update_module._generate_release_index("1.0.0rc1")
    index = (fake_whatsnew / "index.rst").read_text(encoding="utf-8")
    # RC note must appear in the generated index (listing v1.0.0rc1.rst)
    assert "v1.0.0rc1" in index
    assert "v1.0.0" in index


def test_rc_version_keeps_rc_suffix_in_release_notes(update_module, tmp_path):
    # The "latest" note for a release candidate must be named v1.0.0rc1.rst
    notes = {
        "changelog.rst": "What's New in Revision {{ revision }}\n\nsome changes\n",
    }
    for name, content in notes.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    update_module.WN = tmp_path
    update_module.make_release_note_index("1.0.0rc1")
    assert (tmp_path / "v1.0.0rc1.rst").exists()
    assert not (tmp_path / "v1.0.0.rst").exists()


def test_final_release_notes_name(update_module, tmp_path):
    notes = {
        "changelog.rst": "What's New in Revision {{ revision }}\n\nsome changes\n",
    }
    for name, content in notes.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    update_module.WN = tmp_path
    update_module.make_release_note_index("1.0.0")
    assert (tmp_path / "v1.0.0.rst").exists()


def test_shared_helper_release_notes_name():
    from release_version import release_notes_name

    assert release_notes_name("1.0.0rc1") == "v1.0.0rc1.rst"
    assert release_notes_name("1.0.0") == "v1.0.0.rst"
