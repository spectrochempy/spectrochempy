from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "scripts"
    / "evaluate_pr_bypass.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_pr_bypass", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# is_safe_docs_only
# ---------------------------------------------------------------------------


def test_safe_docs_bypass_allows_whatsnew_pair():
    module = _load_module()
    assert module.is_safe_docs_only(
        [
            "docs/sources/whatsnew/changelog.rst",
            "docs/sources/whatsnew/latest.rst",
        ]
    )


def test_safe_docs_bypass_allows_explicit_narrative_docs_allowlist():
    module = _load_module()
    assert module.is_safe_docs_only(
        [
            "docs/sources/credits/citing.rst",
            "docs/sources/reference/faq.rst",
            "docs/sources/reference/bibliography.bib",
        ]
    )


def test_safe_docs_bypass_rejects_other_docs_paths():
    module = _load_module()
    assert not module.is_safe_docs_only(
        [
            "docs/sources/whatsnew/changelog.rst",
            "docs/sources/userguide/plugins_examples.rst",
        ]
    )


def test_safe_docs_bypass_rejects_docs_templates_and_indexes():
    module = _load_module()
    assert not module.is_safe_docs_only(
        [
            "docs/sources/credits/credits.rst.tmpl",
            "docs/sources/index.rst",
        ]
    )


def test_safe_docs_bypass_rejects_gallery_code_changes():
    module = _load_module()
    assert not module.is_safe_docs_only(
        [
            "docs/sources/whatsnew/changelog.rst",
            "src/spectrochempy/examples/core/d_plotting/plot_styles.py",
        ]
    )


def test_safe_docs_bypass_rejects_empty_list():
    """Empty file list → cannot determine safety → must not skip."""
    module = _load_module()
    assert not module.is_safe_docs_only([])


def test_safe_docs_bypass_allows_single_changelog():
    module = _load_module()
    assert module.is_safe_docs_only(["docs/sources/whatsnew/changelog.rst"])


def test_safe_docs_bypass_allows_single_latest():
    module = _load_module()
    assert module.is_safe_docs_only(["docs/sources/whatsnew/latest.rst"])


def test_safe_docs_bypass_allows_mixed_safe_exact_and_docs():
    module = _load_module()
    assert module.is_safe_docs_only(
        [
            "AGENTS.md",
            "CONTRIBUTING.md",
            "docs/sources/whatsnew/changelog.rst",
            "docs/sources/whatsnew/latest.rst",
        ]
    )


def test_safe_docs_bypass_allows_maintainer_markdown():
    module = _load_module()
    assert module.is_safe_docs_only(["maintainers/release-process.md"])


def test_safe_docs_bypass_rejects_maintainer_non_markdown():
    """Maintainers/ prefix is only safe for .md files."""
    module = _load_module()
    assert not module.is_safe_docs_only(["maintainers/some-script.py"])


def test_safe_docs_bypass_rejects_docs_markdown_not_in_allowlist():
    """docs/ .md files not in SAFE_DOCS_EXACT are rejected by UNSAFE_PREFIXES."""
    module = _load_module()
    assert not module.is_safe_docs_only(["docs/sources/random-note.md"])


def test_safe_docs_bypass_allows_unknown_root_markdown():
    """Top-level .md files pass (catch-all at end of chain)."""
    module = _load_module()
    assert module.is_safe_docs_only(["random-note.md"])


def test_safe_docs_bypass_rejects_source_code():
    module = _load_module()
    assert not module.is_safe_docs_only(["src/spectrochempy/core/api.py"])


def test_safe_docs_bypass_rejects_tests():
    module = _load_module()
    assert not module.is_safe_docs_only(["tests/test_core/test_api.py"])


def test_safe_docs_bypass_rejects_examples():
    module = _load_module()
    assert not module.is_safe_docs_only(["examples/plot_1d.py"])


def test_safe_docs_bypass_rejects_plugins():
    module = _load_module()
    assert not module.is_safe_docs_only(["plugins/spectrochempy_nexus/__init__.py"])


def test_safe_docs_bypass_rejects_github_workflows():
    module = _load_module()
    assert not module.is_safe_docs_only([".github/workflows/test_package.yml"])


def test_safe_docs_bypass_rejects_unknown_non_markdown():
    """A file not in any safe list and not .md → rejected."""
    module = _load_module()
    assert not module.is_safe_docs_only(["CHANGELOG.md.bak"])


def test_safe_docs_bypass_allows_all_narrative_docs():
    module = _load_module()
    assert module.is_safe_docs_only(
        [
            "docs/sources/credits/citing.rst",
            "docs/sources/credits/license.rst",
            "docs/sources/credits/seealso.rst",
            "docs/sources/gettingstarted/getting_help.rst",
            "docs/sources/gettingstarted/whyscpy.rst",
            "docs/sources/reference/bibliography.bib",
            "docs/sources/reference/bibliography.rst",
            "docs/sources/reference/faq.rst",
            "docs/sources/reference/glossary.rst",
            "docs/sources/reference/papers.rst",
            "docs/sources/whatsnew/changelog.rst",
            "docs/sources/whatsnew/latest.rst",
        ]
    )


# ---------------------------------------------------------------------------
# _is_zero_sha
# ---------------------------------------------------------------------------


def test_is_zero_sha_all_zeros():
    module = _load_module()
    assert module._is_zero_sha("0000000000000000000000000000000000000000")


def test_is_zero_sha_short_zeros():
    module = _load_module()
    assert module._is_zero_sha("0000000")


def test_is_zero_sha_single_zero():
    module = _load_module()
    assert module._is_zero_sha("0")


def test_is_zero_sha_empty():
    module = _load_module()
    assert not module._is_zero_sha("")


def test_is_zero_sha_none():
    module = _load_module()
    assert not module._is_zero_sha("None")


def test_is_zero_sha_normal_sha():
    module = _load_module()
    assert not module._is_zero_sha("abc123def456")


def test_is_zero_sha_mixed_zeros_and_hex():
    module = _load_module()
    assert not module._is_zero_sha("0a0b0c0d")


# ---------------------------------------------------------------------------
# changed_files — mocked git
# ---------------------------------------------------------------------------


def test_changed_files_normal_three_dot():
    module = _load_module()
    with patch.object(
        module, "_run_git", return_value=["src/api.py", "tests/test_api.py"]
    ) as mock:
        result = module.changed_files("abc123", "def456")
    assert result == ["src/api.py", "tests/test_api.py"]
    mock.assert_called_once_with(["diff", "--name-only", "abc123...def456"])


def test_changed_files_fallback_to_two_dot():
    """If three-dot fails, try two-dot."""
    module = _load_module()
    call_count = 0

    def mock_run_git(args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise subprocess.CalledProcessError(1, "git")
        return ["file.txt"]

    with patch.object(module, "_run_git", side_effect=mock_run_git) as mock:
        result = module.changed_files("abc123", "def456")
    assert result == ["file.txt"]
    assert mock.call_count == 2
    mock.assert_any_call(["diff", "--name-only", "abc123...def456"])
    mock.assert_any_call(["diff", "--name-only", "abc123..def456"])


def test_changed_files_both_dots_fail_fallback_to_head():
    """If both three-dot and two-dot fail, fall back to HEAD~1..HEAD."""
    module = _load_module()
    call_count = 0

    def mock_run_git(args):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise subprocess.CalledProcessError(1, "git")
        return ["fallback.py"]

    with patch.object(module, "_run_git", side_effect=mock_run_git) as mock:
        result = module.changed_files("abc123", "def456")
    assert result == ["fallback.py"]
    assert mock.call_count == 3
    mock.assert_any_call(["diff", "--name-only", "HEAD~1..HEAD"])


def test_changed_files_zero_sha_falls_back_to_head():
    """Zero SHA as base → skip to HEAD~1..HEAD fallback."""
    module = _load_module()
    with patch.object(module, "_run_git", return_value=["x.py"]) as mock:
        result = module.changed_files("0000000", "abc123")
    assert result == ["x.py"]
    mock.assert_called_once_with(["diff", "--name-only", "HEAD~1..HEAD"])


def test_changed_files_empty_base_falls_back_to_head():
    """Empty string base → skip to HEAD~1..HEAD fallback."""
    module = _load_module()
    with patch.object(module, "_run_git", return_value=["y.py"]) as mock:
        result = module.changed_files("", "abc123")
    assert result == ["y.py"]
    mock.assert_called_once_with(["diff", "--name-only", "HEAD~1..HEAD"])


def test_changed_files_none_base_falls_back_to_head():
    """None base → skip to HEAD~1..HEAD fallback."""
    module = _load_module()
    with patch.object(module, "_run_git", return_value=["z.py"]) as mock:
        result = module.changed_files(None, "abc123")
    assert result == ["z.py"]
    mock.assert_called_once_with(["diff", "--name-only", "HEAD~1..HEAD"])


def test_changed_files_none_head_defaults_to_head():
    """None head → defaults to HEAD."""
    module = _load_module()
    with patch.object(module, "_run_git", return_value=["a.py"]) as mock:
        module.changed_files("abc123", None)
    mock.assert_called_once_with(["diff", "--name-only", "abc123...HEAD"])


def test_changed_files_all_fallbacks_fail_returns_empty():
    """If every git command fails → return empty list (safe fallback)."""
    module = _load_module()
    with patch.object(
        module, "_run_git", side_effect=subprocess.CalledProcessError(1, "git")
    ):
        result = module.changed_files("abc123", "def456")
    assert result == []


def test_changed_files_empty_diff_returns_empty():
    """No changed files → empty list → is_safe_docs_only will return False."""
    module = _load_module()
    with patch.object(module, "_run_git", return_value=[]):
        result = module.changed_files("abc123", "def456")
    assert result == []


# ---------------------------------------------------------------------------
# main — end-to-end label + file classification
# ---------------------------------------------------------------------------


def test_main_skip_when_label_and_safe_files():
    module = _load_module()
    files = ["docs/sources/whatsnew/changelog.rst", "docs/sources/whatsnew/latest.rst"]
    labels = {"safe-docs-no-ci"}
    assert module.SAFE_DOCS_LABEL in labels
    assert module.is_safe_docs_only(files)


def test_main_no_skip_when_label_missing():
    module = _load_module()
    files = ["docs/sources/whatsnew/changelog.rst"]
    labels = set()
    has_label = module.SAFE_DOCS_LABEL in labels
    safe_only = module.is_safe_docs_only(files)
    should_skip = has_label and safe_only
    assert not should_skip


def test_main_no_skip_when_unsafe_file_present():
    module = _load_module()
    files = ["docs/sources/whatsnew/changelog.rst", "src/spectrochempy/api.py"]
    labels = {"safe-docs-no-ci"}
    has_label = module.SAFE_DOCS_LABEL in labels
    safe_only = module.is_safe_docs_only(files)
    should_skip = has_label and safe_only
    assert not should_skip


def test_main_no_skip_when_empty_files():
    module = _load_module()
    files: list[str] = []
    labels = {"safe-docs-no-ci"}
    has_label = module.SAFE_DOCS_LABEL in labels
    safe_only = module.is_safe_docs_only(files)
    should_skip = has_label and safe_only
    assert not should_skip
    assert not module.is_safe_docs_only(files)
