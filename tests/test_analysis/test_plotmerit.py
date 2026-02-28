# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for plotmerit functionality including multi-regularization support."""

import numpy as np
import pytest


def test_normalize_xy_for_render_1d_to_1d():
    """Test case 1: x (n,), y (n,) -> x (n,), y (1, n)."""
    from spectrochempy.plotting.composite.plotmerit import _normalize_xy_for_render

    x = np.arange(10)
    y = np.arange(10) * 2

    x_out, y_out = _normalize_xy_for_render(x, y)

    assert x_out.shape == (10,)
    assert y_out.shape == (1, 10)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out[0], y)


def test_normalize_xy_for_render_1d_to_2d_transposed():
    """Test case 2: x (n,), y (n, k) -> transpose to y (k, n)."""
    from spectrochempy.plotting.composite.plotmerit import _normalize_xy_for_render

    n, k = 10, 5
    x = np.arange(n)
    y = np.random.rand(n, k)

    x_out, y_out = _normalize_xy_for_render(x, y)

    assert x_out.shape == (n,)
    assert y_out.shape == (k, n)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y.T)


def test_normalize_xy_for_render_1d_to_2d_standard():
    """Test case 3: x (n,), y (k, n) -> unchanged."""
    from spectrochempy.plotting.composite.plotmerit import _normalize_xy_for_render

    n, k = 10, 5
    x = np.arange(n)
    y = np.random.rand(k, n)

    x_out, y_out = _normalize_xy_for_render(x, y)

    assert x_out.shape == (n,)
    assert y_out.shape == (k, n)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y)


def test_normalize_xy_for_render_per_line_x():
    """Test case 4: x (k,), y (k, n) -> transpose when x doesn't match."""
    from spectrochempy.plotting.composite.plotmerit import _normalize_xy_for_render

    n, k = 10, 5
    x = np.arange(k)
    y = np.random.rand(k, n)

    x_out, y_out = _normalize_xy_for_render(x, y)
    assert x_out.shape == (k,)
    assert y_out.shape == (n, k)
    np.testing.assert_array_equal(y_out, y.T)


def test_normalize_xy_for_render_length_mismatch():
    """Test length mismatch raises ValueError."""
    from spectrochempy.plotting.composite.plotmerit import _normalize_xy_for_render

    x = np.arange(10)
    y = np.arange(20)

    with pytest.raises(ValueError, match="Length mismatch"):
        _normalize_xy_for_render(x, y)


def test_plotmerit_single_index():
    """Test plotmerit with a single index works correctly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import spectrochempy as scp

    X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    X = X[:, 2250.0:1950.0]
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    c_pressures = scp.Coord(pressures, title="pressure", units="torr")
    c_times = X.y.copy()
    X.y = [c_times, c_pressures]
    X.y.select(2)

    K = scp.IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = scp.IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)

    ax = iris.plotmerit(index=0, show=False)

    n_traces = X.shape[0]
    expected_lines = 3 * n_traces  # resid, orig, recon
    assert len(ax.lines) == expected_lines

    plt.close()


def test_plotmerit_multi_index():
    """Test plotmerit with multiple indices works correctly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import spectrochempy as scp

    X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    X = X[:, 2250.0:1950.0]
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    c_pressures = scp.Coord(pressures, title="pressure", units="torr")
    c_times = X.y.copy()
    X.y = [c_times, c_pressures]
    X.y.select(2)

    K = scp.IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = scp.IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)

    axes_list = iris.plotmerit(index=[0, 1, 2], show=False)

    assert len(axes_list) == 3
    for ax in axes_list:
        n_traces = X.shape[0]
        expected_lines = 3 * n_traces
        assert len(ax.lines) == expected_lines
        plt.close()


def test_plotmerit_all_regularizations():
    """Test plotmerit with index=None (all regularizations on same axes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import spectrochempy as scp

    X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    X = X[:, 2250.0:1950.0]
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    c_pressures = scp.Coord(pressures, title="pressure", units="torr")
    c_times = X.y.copy()
    X.y = [c_times, c_pressures]
    X.y.select(2)

    K = scp.IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = scp.IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)

    ax = iris.plotmerit(show=False)

    n_reg = 3
    n_traces = X.shape[0]
    expected_lines = n_reg * 3 * n_traces  # 3 regularizations * 3 lines per trace
    assert len(ax.lines) == expected_lines

    plt.close()


def test_plotmerit_y_limits():
    """Test that y-limits ensure residuals are visible and top is tight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import spectrochempy as scp

    X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    X = X[:, 2250.0:1950.0]
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    c_pressures = scp.Coord(pressures, title="pressure", units="torr")
    c_times = X.y.copy()
    X.y = [c_times, c_pressures]
    X.y.select(2)

    K = scp.IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = scp.IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)

    ax = iris.plotmerit(index=0, show=False)
    ylim = ax.get_ylim()
    y_min, y_max = ylim

    _n_traces = X.shape[0]
    _n_points = X.shape[1]

    X_hat = iris.inverse_transform()
    X_hat_0 = X_hat[0].squeeze()

    ma = max(X.max(), X_hat_0.max())
    mad = ma * 0 / 100 + ma / 10

    res = X - X_hat_0
    res_offset = res - mad
    res_data = np.asarray(res_offset.masked_data)
    exp_data = np.asarray(X.masked_data)
    recon_data = np.asarray(X_hat_0.masked_data)

    res_min = np.nanmin(res_data)
    main_max = max(np.nanmax(exp_data), np.nanmax(recon_data))

    assert y_min < res_min, f"y_min ({y_min}) should be below residual min ({res_min})"
    assert y_max >= main_max, f"y_max ({y_max}) should be >= main max ({main_max})"

    data_range = main_max - res_min
    if data_range > 0:
        expected_pad = 0.02 * data_range
    else:
        expected_pad = 0.02 * max(1.0, abs(main_max), abs(res_min))

    assert y_max <= main_max + expected_pad * 1.1, "y_max has too much padding"

    plt.close()


def test_plotmerit_zorder():
    """Test zorder layering: residuals (0) < reconstructed (1) < experimental (2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import spectrochempy as scp

    X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    X = X[:, 2250.0:1950.0]
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    c_pressures = scp.Coord(pressures, title="pressure", units="torr")
    c_times = X.y.copy()
    X.y = [c_times, c_pressures]
    X.y.select(2)

    K = scp.IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = scp.IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)

    ax = iris.plotmerit(index=0, show=False)

    n_traces = X.shape[0]

    res_zorders = [ax.lines[i].get_zorder() for i in range(n_traces)]
    exp_zorders = [ax.lines[i].get_zorder() for i in range(n_traces, 2 * n_traces)]
    recon_zorders = [
        ax.lines[i].get_zorder() for i in range(2 * n_traces, 3 * n_traces)
    ]

    assert all(z == 0 for z in res_zorders), "All residual zorders should be 0"
    assert all(z == 1 for z in recon_zorders), "All reconstructed zorders should be 1"
    assert all(z == 2 for z in exp_zorders), "All experimental zorders should be 2"

    assert max(res_zorders) < min(
        recon_zorders
    ), "Residuals should be behind reconstructed"
    assert max(recon_zorders) < min(
        exp_zorders
    ), "Reconstructed should be behind experimental"

    plt.close()
