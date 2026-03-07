# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for plotmerit functionality including multi-regularization support."""

import numpy as np


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

    ax = iris.plot_merit(index=0, show=False)

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

    axes_list = iris.plot_merit(index=[0, 1, 2], show=False)

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

    ax = iris.plot_merit(show=False)

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

    ax = iris.plot_merit(index=0, show=False)
    ylim = ax.get_ylim()
    y_min, y_max = ylim

    _n_traces = X.shape[0]
    _n_points = X.shape[1]

    X_hat = iris.inverse_transform()
    X_hat_0 = X_hat[0].squeeze()

    ma = max(X.max(), X_hat_0.max())
    mad = ma * 0 / 100 + ma / 10

    res = X - X_hat_0
    # Note: plot_merit does NOT apply offset to residuals, uses raw residuals
    res_data = np.asarray(res.masked_data)
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

    ax = iris.plot_merit(index=0, show=False)

    n_traces = X.shape[0]

    # Note: lines are added in order: residual (z=0), reconstructed (z=1), experimental (z=2)
    # So ax.lines indices are: 0:n = residual, n:2n = reconstructed, 2n:3n = experimental
    res_zorders = [ax.lines[i].get_zorder() for i in range(n_traces)]
    recon_zorders = [ax.lines[i].get_zorder() for i in range(n_traces, 2 * n_traces)]
    exp_zorders = [ax.lines[i].get_zorder() for i in range(2 * n_traces, 3 * n_traces)]

    assert all(z == 0 for z in res_zorders), "All residual zorders should be 0"
    assert all(z == 1 for z in recon_zorders), "All reconstructed zorders should be 1"
    assert all(z == 2 for z in exp_zorders), "All experimental zorders should be 2"

    assert max(res_zorders) < min(recon_zorders), (
        "Residuals should be behind reconstructed"
    )
    assert max(recon_zorders) < min(exp_zorders), (
        "Reconstructed should be behind experimental"
    )

    plt.close()
