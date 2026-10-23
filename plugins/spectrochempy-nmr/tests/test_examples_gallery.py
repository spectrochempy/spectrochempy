# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, S603

"""Targeted checks for the public NMR example gallery."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    "processing/nmr/plot_exam2d_hh_qf_cosy.py",
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


def test_nmr_gallery_manifest_focuses_on_public_examples():
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

    result = subprocess.run(
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


@pytest.fixture(scope="module")
def cosy_example():
    root = Path(scp.preferences.datadir) / "nmrdata/bruker/tests/nmr/exam2d_HH/1"
    if not (root / "ser").exists() or not (root / "pdata/1/2rr").exists():
        pytest.skip("Bundled COSY reference data not available")

    figures_before = set(plt.get_fignums())
    try:
        yield runpy.run_path(
            str(PLUGIN_ROOT / "examples/processing/nmr/plot_exam2d_hh_qf_cosy.py")
        )
    finally:
        for number in set(plt.get_fignums()) - figures_before:
            plt.close(number)


@pytest.mark.filterwarnings(
    "ignore:the matrix subclass is not the recommended way:PendingDeprecationWarning"
)
def test_cosy_diagnostics_preserve_global_scale_and_physical_diagonal(cosy_example):
    example = cosy_example
    residual = example["residual"].data
    displayed_rms = np.linalg.norm(residual) / np.linalg.norm(example["topspin"].data)
    assert displayed_rms == pytest.approx(example["rms_rel"], rel=1e-12)
    assert np.max(example["topspin"].data) == pytest.approx(1.0)
    np.testing.assert_allclose(
        (example["replayed"] * example["reference_max"]).data,
        example["replayed_scaled"].data,
    )

    x = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    y = np.array([3.5, 2.0, 0.5])
    values = 2.0 * y[:, None] + 3.0 * x[None, :]
    field = scp.NDDataset(values, coordset=[scp.Coord(y), scp.Coord(x)])
    profile = example["_diagonal_cut"](field)
    np.testing.assert_array_equal(profile.x.data, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(profile.data, 5.0 * profile.x.data)

    cut_y, cut_x = example["cut_y"], example["cut_x"]
    assert example["off_diagonal"][cut_y, cut_x]
    assert abs(residual[cut_y, cut_x]) == np.max(
        np.abs(residual)[example["off_diagonal"]]
    )


@pytest.mark.filterwarnings(
    "ignore:the matrix subclass is not the recommended way:PendingDeprecationWarning"
)
def test_cosy_native_processing_preserves_bounded_replay(cosy_example):
    example = cosy_example
    raw = example["raw"]
    untouched = scp.nmr.read(example["dataset"], expno=1, remove_digital_filter=True)
    np.testing.assert_array_equal(raw.data, untouched.data)
    np.testing.assert_array_equal(raw.x.data, untouched.x.data)
    np.testing.assert_array_equal(raw.y.data, untouched.y.data)
    assert raw.meta == untouched.meta

    size = example["SI"]
    n = raw.shape[1]
    windowed = raw.data * np.sin(np.pi * np.arange(n) / (n - 1))
    f2 = np.fft.fftshift(np.fft.fft(windowed, n=size, axis=1), axes=1)
    f2[0, :] *= 0.5
    lp_size = example["LPBIN_F1"]
    predicted = np.column_stack(
        [
            example["_lp_predict_after"](f2[:, i], lp_size, example["NCOEF_F1"])
            for i in range(size)
        ]
    )
    predicted *= np.sin(np.pi * np.arange(lp_size) / (lp_size - 1))[:, None]
    expected = np.abs(
        np.fft.fftshift(np.fft.fft(predicted.conj(), n=size, axis=0), axes=0)
    )
    result = example["replayed_spectrum"]
    assert isinstance(result, scp.NDDataset)
    assert result.shape == (size, size)
    assert result.dims == ["y", "x"]
    assert result.meta.encoding == ["QF", "DQD"]
    assert result.meta.isfreq == [True, True]
    np.testing.assert_allclose(
        result.data, expected, rtol=1e-7, atol=expected.max() * 1e-7
    )
    assert np.linalg.norm(result.data - expected) / np.linalg.norm(expected) < 1e-7
    np.testing.assert_array_equal(result.data, example["replayed_raw"].data)
    np.testing.assert_array_equal(
        example["replayed_raw"].x.data, example["topspin_2rr"].x.data
    )
    assert not np.array_equal(result.x.data, example["replayed_raw"].x.data)
