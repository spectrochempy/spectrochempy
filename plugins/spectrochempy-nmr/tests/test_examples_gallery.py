# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Targeted checks for the public NMR example gallery."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import spectrochempy as scp


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "spectrochempy-nmr"
MANIFEST = PLUGIN_ROOT / "examples" / "gallery.toml"
VISIBLE_EXAMPLES = [
    "core/c_importer/plot_read_nmr_from_bruker.py",
    "processing/apodization/plot_proc_em.py",
    "processing/apodization/plot_proc_sp.py",
    "processing/nmr/plot_read_nmr_topspin.py",
    "processing/nmr/plot_processing_nmr.py",
    "processing/nmr/plot_processing_nmr_relax.py",
]


def _has_nmr_example_data() -> bool:
    root = Path(scp.preferences.datadir) / "nmrdata" / "bruker" / "tests" / "nmr"
    required = [
        root / "topspin_1d" / "1" / "fid",
        root / "relax" / "100" / "ser",
    ]
    return all(path.exists() for path in required)


def test_nmr_gallery_manifest_focuses_on_public_1d_examples():
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    example_paths = [entry["path"] for entry in manifest["examples"]]

    assert example_paths == VISIBLE_EXAMPLES
    assert "processing/nmr/plot_processing_cp_nmr.py" not in example_paths
    assert "processing/nmr/plot_processing_nmr_2d.py" not in example_paths


@pytest.mark.parametrize("relative_path", VISIBLE_EXAMPLES, ids=VISIBLE_EXAMPLES)
def test_public_nmr_gallery_examples_execute(relative_path, tmp_path):
    if not _has_nmr_example_data():
        pytest.skip("Bundled NMR example data not available in this environment")

    example = PLUGIN_ROOT / "examples" / relative_path

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["SCP_CONFIG_HOME"] = str(tmp_path / "scp-config")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "src"),
            str(PLUGIN_ROOT / "src"),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(example)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    error_msg = (
        f"Example failed: {relative_path}\n"
        f"Return code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    assert result.returncode == 0, error_msg
