# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for transient MCRALS config persistence behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from collections import Counter
import io
import threading

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.application.application import app
from spectrochempy.application.config import SpectroChemPyJSONConfigManager


class CountingConfigManager(SpectroChemPyJSONConfigManager):
    """Count config persistence writes while keeping real JSON behavior."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self._calls_lock = threading.Lock()

    def update(self, section_name, new_data):
        with self._calls_lock:
            self.calls.append((section_name, tuple(new_data.get(section_name, {}).keys())))
        return super().update(section_name, new_data)


@pytest.fixture
def counting_manager(tmp_path):
    previous = app.config_manager
    manager = CountingConfigManager(config_dir=str(tmp_path))
    app.config_manager = manager
    try:
        yield manager
    finally:
        app.config_manager = previous


@pytest.fixture
def simple_peak():
    n_mz = 20
    n_components = 2
    n_scans = 30
    rng = np.random.default_rng(0)
    t = np.arange(n_scans)
    centers = np.linspace(0.25, 0.75, n_components) * n_scans
    C = np.stack(
        [np.exp(-((t - c) ** 2) / (n_scans / 3.0)) for c in centers],
        axis=1,
    )
    S = np.abs(rng.normal(size=(n_components, n_mz))) + 0.1
    X = scp.NDDataset(C @ S, name="peak", title="intensity")
    X.set_coordset(
        y=scp.Coord(t.astype(float), title="retention time", units="s"),
        x=scp.Coord(np.arange(n_mz), title="m/z"),
    )
    return X, scp.NDDataset(C.copy())


def test_fit_revalidates_component_traits_without_persisting_internal_writes(
    counting_manager,
    simple_peak,
):
    X, C0 = simple_peak
    mcr = MCRALS(max_iter=5, tol=1.0, log_level="WARNING")

    assert [trait for _, traits in counting_manager.calls for trait in traits] == [
        "max_iter",
        "tol",
        "tol_residual_change",
    ]

    counting_manager.calls.clear()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        mcr.fit(X, C0)

    assert stdout.getvalue() == ""
    assert counting_manager.calls == []
    assert np.array_equal(mcr.closureTarget, np.ones(X.shape[0]))
    assert mcr.getC_to_C_idx == [0, 1]
    assert mcr.getSt_to_St_idx == [0, 1]
    assert mcr.nonnegConc == [0, 1]
    assert mcr.nonnegSpec == [0, 1]
    assert mcr.unimodConc == [0, 1]
    assert mcr.unimodSpec == []
    assert mcr.closureConc == []


def test_fit_preserves_validation_failures_for_component_dependent_traits(
    counting_manager,
    simple_peak,
):
    X, C0 = simple_peak
    mcr = MCRALS(max_iter=5, tol=1.0, log_level="WARNING", nonnegConc=[0, 1, 2])

    with pytest.raises(ValueError, match="nonnegConc"):
        mcr.fit(X, C0)


def test_fit_preserves_valid_explicit_component_dependent_values(
    counting_manager,
    simple_peak,
):
    X, C0 = simple_peak
    target = np.linspace(1.0, 2.0, X.shape[0]).tolist()
    mcr = MCRALS(
        max_iter=5,
        tol=1.0,
        log_level="WARNING",
        nonnegConc=[1],
        nonnegSpec=[0],
        unimodConc=[1],
        unimodSpec=[0],
        closureConc=[0],
        closureTarget=target,
        getC_to_C_idx=[1, 0],
        getSt_to_St_idx=[1, 0],
    )

    counting_manager.calls.clear()
    mcr.fit(X, C0)

    assert counting_manager.calls == []
    assert mcr.nonnegConc == [1]
    assert mcr.nonnegSpec == [0]
    assert mcr.unimodConc == [1]
    assert mcr.unimodSpec == [0]
    assert mcr.closureConc == [0]
    assert np.array_equal(mcr.closureTarget, np.asarray(target))
    assert mcr.getC_to_C_idx == [1, 0]
    assert mcr.getSt_to_St_idx == [1, 0]


def test_explicit_user_config_change_still_persists(counting_manager):
    mcr = MCRALS(max_iter=5, tol=1.0, log_level="WARNING")
    counting_manager.calls.clear()

    mcr.max_iter = 7

    assert counting_manager.calls == [("MCRALS", ("max_iter",))]


def test_concurrent_fits_only_persist_constructor_writes(
    counting_manager,
):
    peaks = []
    for n_scans, seed in ((30, 1), (18, 2)):
        rng = np.random.default_rng(seed)
        n_mz = 20
        n_components = 2
        t = np.arange(n_scans)
        centers = np.linspace(0.25, 0.75, n_components) * n_scans
        C = np.stack(
            [np.exp(-((t - c) ** 2) / (n_scans / 3.0)) for c in centers],
            axis=1,
        )
        S = np.abs(rng.normal(size=(n_components, n_mz))) + 0.1
        X = scp.NDDataset(C @ S, name=f"peak{n_scans}", title="intensity")
        X.set_coordset(
            y=scp.Coord(t.astype(float), title="retention time", units="s"),
            x=scp.Coord(np.arange(n_mz), title="m/z"),
        )
        peaks.append((X, scp.NDDataset(C.copy())))

    counting_manager.calls.clear()

    def fit_one(peak):
        X, C0 = peak
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            mcr = MCRALS(max_iter=5, tol=1.0, log_level="WARNING")
            mcr.fit(X, C0)
        return stdout.getvalue(), mcr._fitted, mcr.C.shape, mcr.St.shape

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fit_one, peak) for peak in peaks]
        results = [future.result(timeout=10) for future in futures]

    assert all(stdout == "" for stdout, _, _, _ in results)
    assert all(fitted for _, fitted, _, _ in results)
    assert all(c_shape[1] == 2 and st_shape[0] == 2 for _, _, c_shape, st_shape in results)
    assert Counter(
        trait for _, traits in counting_manager.calls for trait in traits
    ) == Counter(
        {
            "max_iter": 2,
            "tol": 2,
            "tol_residual_change": 2,
        }
    )
