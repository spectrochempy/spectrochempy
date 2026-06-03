# ======================================================================================
# Copyright (C) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Regression tests for the NMR plugin importer handlers."""

import pytest

pytestmark = pytest.mark.plugin

pytest.importorskip(
    "spectrochempy_nmr",
    reason="requires the optional spectrochempy-nmr plugin",
)

from pathlib import Path

from spectrochempy_nmr import _resolve_topspin_directory_target
from spectrochempy_nmr import _topspin_remote_download_target


def _norm(path: str) -> str:
    """Normalise un chemin retourné par les handlers pour comparaison portable."""
    return path.rstrip("/\\").replace("\\", "/")


def test_missing_topspin_directory_resolves_remote_default_experiment():
    directory = Path("/missing/topspin_1d")

    resolved = _resolve_topspin_directory_target(directory, protocol=["topspin"])

    assert resolved == [directory / "1" / "fid"]
    target = _topspin_remote_download_target(resolved[0], protocol=["topspin"])
    assert _norm(target) == "/missing/topspin_1d/1"


def test_missing_topspin_directory_preserves_expno_for_remote_target():
    directory = Path("/missing/h3po4")

    resolved = _resolve_topspin_directory_target(
        directory,
        protocol=["topspin"],
        expno=4,
    )

    assert resolved == [directory / "4" / "fid"]


def test_missing_explicit_topspin_file_is_not_resolved_as_directory():
    filename = Path("/missing/topspin_1d/1/fid")

    assert _resolve_topspin_directory_target(filename, protocol=["topspin"]) is None
    target = _topspin_remote_download_target(filename, protocol=["topspin"])
    assert _norm(target) == "/missing/topspin_1d/1"
