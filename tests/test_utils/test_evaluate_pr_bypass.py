from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_safe_docs_bypass_allows_whatsnew_pair_only():
    module = _load_module()

    assert module.is_safe_docs_only(
        [
            "docs/sources/whatsnew/changelog.rst",
            "docs/sources/whatsnew/latest.rst",
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


def test_safe_docs_bypass_rejects_gallery_code_changes():
    module = _load_module()

    assert not module.is_safe_docs_only(
        [
            "docs/sources/whatsnew/changelog.rst",
            "src/spectrochempy/examples/core/d_plotting/plot_styles.py",
        ]
    )
