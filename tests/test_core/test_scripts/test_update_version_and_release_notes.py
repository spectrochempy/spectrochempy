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
INDEX_TEMPLATE = Path(__file__).parents[3] / "docs" / "sources" / "index.rst.tmpl"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("setuptools_scm") is None,
    reason="setuptools_scm not available in this environment",
)

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


def test_release_index_is_orphan(update_module, fake_whatsnew):
    update_module._generate_release_index("1.0.0rc1")
    index = (fake_whatsnew / "index.rst").read_text(encoding="utf-8")

    assert index.startswith(":orphan:\n\n.. _release:")


def test_current_release_is_top_level_navigation():
    index_template = INDEX_TEMPLATE.read_text(encoding="utf-8")

    assert "\n    whatsnew/latest\n" in index_template
    assert "\n    whatsnew/index\n" not in index_template


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


def test_final_release_consolidates_release_candidate_notes(update_module, tmp_path):
    update_module.WN = tmp_path

    def changelog(change):
        return update_module._get_changelog_template().replace(
            ".. Add here new bug fixes (do not delete this comment)",
            f".. Add here new bug fixes (do not delete this comment)\n\n- {change}",
        )

    (tmp_path / "changelog.rst").write_text(
        changelog("Fixed the first candidate issue."),
        encoding="utf-8",
    )
    update_module.make_release_note_index("1.0.0rc1")

    (tmp_path / "changelog.rst").write_text(
        changelog("Fixed the second candidate issue."),
        encoding="utf-8",
    )
    update_module.make_release_note_index("1.0.0rc2")

    # Preparing the final release immediately after RC2 starts from the empty
    # changelog template, but its notes must still describe the complete cycle.
    update_module.make_release_note_index("1.0.0")

    final = (tmp_path / "v1.0.0.rst").read_text(encoding="utf-8")
    assert "Fixed the first candidate issue." in final
    assert "Fixed the second candidate issue." in final
    assert "``1.0.0rc1``, ``1.0.0rc2``" in final
    assert final.count("Bug Fixes\n~~~~~~~~~") == 1

    assert "Fixed the first candidate issue." in (tmp_path / "v1.0.0rc1.rst").read_text(
        encoding="utf-8"
    )
    assert "Fixed the second candidate issue." in (
        tmp_path / "v1.0.0rc2.rst"
    ).read_text(encoding="utf-8")


def test_shared_helper_release_notes_name():
    from release_version import release_notes_name

    assert release_notes_name("1.0.0rc1") == "v1.0.0rc1.rst"
    assert release_notes_name("1.0.0") == "v1.0.0.rst"


@pytest.mark.parametrize(
    ("gitversion", "prepared", "expected"),
    [
        ("0.12.9.dev0", "1.0.0rc1", "1.0.0rc2.dev"),
        ("1.0.0rc2.dev4", "1.0.0rc1", "1.0.0rc2.dev"),
        ("1.0.0rc3.dev2", "1.0.0rc1", "1.0.0rc3.dev"),
        ("1.0.0rc2.dev0", "1.0.0", "1.0.1.dev"),
        ("0.12.9.dev0", "0.12.8", "0.12.9.dev"),
    ],
)
def test_unreleased_notes_follow_prepared_release(
    update_module, tmp_path, monkeypatch, gitversion, prepared, expected
):
    monkeypatch.setattr(update_module, "WN", tmp_path)
    monkeypatch.setattr(update_module, "gitversion", gitversion)
    changelog = "What's New in Revision {{ revision }}\n\npending fix\n"
    (tmp_path / "changelog.rst").write_text(changelog, encoding="utf-8")
    release_note = tmp_path / f"v{prepared}.rst"
    release_note.write_text("frozen release notes\n", encoding="utf-8")
    (tmp_path / "v0.1.0.rst").write_text("older release\n", encoding="utf-8")

    update_module.make_release_note_index("unreleased")
    latest = (tmp_path / "latest.rst").read_text(encoding="utf-8")
    index = (tmp_path / "index.rst").read_text(encoding="utf-8")
    assert f"What's New in Revision {expected}\n" in latest
    assert "pending fix" in latest
    assert "    latest\n" in index
    assert release_note.read_text(encoding="utf-8") == "frozen release notes\n"
    assert (tmp_path / "changelog.rst").read_text(encoding="utf-8") == changelog

    update_module.make_release_note_index("unreleased")
    assert (tmp_path / "latest.rst").read_text(encoding="utf-8") == latest
    assert (tmp_path / "index.rst").read_text(encoding="utf-8") == index
