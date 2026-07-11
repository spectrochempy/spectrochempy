# ruff: noqa: S101, PLC0415
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Functional tests for plot_iris_lcurve and plot_iris_distribution."""

import pytest

pytestmark = [pytest.mark.plugin, pytest.mark.data]

pytest.importorskip(
    "spectrochempy_iris",
    reason="requires the optional spectrochempy-iris plugin",
)


def _fitted_iris():
    """Return a fitted IRIS object for testing."""
    import matplotlib

    matplotlib.use("Agg")

    from spectrochempy_iris import IRIS
    from spectrochempy_iris import IrisKernel

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

    K = IrisKernel(X, "langmuir", q=[-8, -1, 50])
    iris = IRIS(reg_par=[-10, 1, 3])
    iris.fit(X, K)
    return iris


class TestPlotIrisLcurve:
    """Tests for plot_iris_lcurve."""

    @staticmethod
    def test_returns_axes():
        """plot_iris_lcurve returns an Axes object."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(show=False)
        assert hasattr(ax, "plot")
        plt.close()

    @staticmethod
    def test_title_default():
        """Default title is 'L curve'."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(show=False)
        assert ax.get_title() == "L curve"
        plt.close()

    @staticmethod
    def test_title_custom():
        """Custom title is applied."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(title="Custom L curve", show=False)
        assert ax.get_title() == "Custom L curve"
        plt.close()

    @staticmethod
    def test_scale_ll():
        """Default 'll' sets both axes to log scale."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(show=False)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close()

    @staticmethod
    def test_scale_nn():
        """'nn' sets both axes to linear scale."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(scale="nn", show=False)
        assert ax.get_xscale() == "linear"
        assert ax.get_yscale() == "linear"
        plt.close()

    @staticmethod
    def test_provided_ax():
        """Provided ax is reused."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        fig, ax = plt.subplots()
        result_ax = iris.plotlcurve(ax=ax, show=False)
        assert result_ax is ax
        plt.close()

    @staticmethod
    def test_provided_ax_clear_false():
        """With clear=False, existing artists are preserved."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        fig, ax = plt.subplots()
        iris.plotlcurve(ax=ax, show=False)
        n_artists_before = len(ax.collections)

        iris.plotlcurve(ax=ax, show=False, clear=False)
        n_artists_after = len(ax.collections)
        assert n_artists_after > n_artists_before
        plt.close()

    @staticmethod
    def test_show_false_does_not_display():
        """show=False must not trigger display."""
        iris = _fitted_iris()
        ax = iris.plotlcurve(show=False)
        assert ax is not None

    @staticmethod
    def test_labels():
        """Axes labels are set correctly."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(show=False)
        assert ax.get_xlabel() == "Residuals"
        assert ax.get_ylabel() == "Curvature"
        plt.close()

    @staticmethod
    def test_marker_custom():
        """Custom marker is passed through to scatter."""
        import matplotlib.pyplot as plt

        iris = _fitted_iris()
        ax = iris.plotlcurve(marker="x", show=False)
        assert ax is not None
        plt.close()

    @staticmethod
    def test_not_fitted_error():
        """NotFittedError is raised when IRIS is not fitted."""
        from spectrochempy_iris import IRIS

        from spectrochempy.utils.exceptions import NotFittedError

        iris = IRIS()
        with pytest.raises(NotFittedError):
            iris.plotlcurve(show=False)


class TestPlotIrisDistribution:
    """Tests for plot_iris_distribution."""

    @staticmethod
    def test_returns_axes_single_index():
        """Single index returns a single Axes."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        ax = plot_iris_distribution(iris, index=0, show=False)
        assert hasattr(ax, "plot")
        plt.close()

    @staticmethod
    def test_returns_list_multi_index():
        """Multiple indices return a list of Axes."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        axes = plot_iris_distribution(iris, index=[0, 1], show=False)
        assert isinstance(axes, list)
        assert len(axes) == 2
        for ax in axes:
            assert hasattr(ax, "plot")
        plt.close("all")

    @staticmethod
    def test_provided_ax_single_index():
        """Single index with provided ax reuses it."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        fig, ax = plt.subplots()
        result_ax = plot_iris_distribution(iris, index=0, ax=ax, show=False)
        assert result_ax is ax
        plt.close()

    @staticmethod
    def test_custom_title():
        """Custom title is applied."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        ax = plot_iris_distribution(iris, index=0, title="Custom title", show=False)
        assert ax.get_title() == "Custom title"
        plt.close()

    @staticmethod
    def test_show_false_does_not_display():
        """show=False must not trigger display."""
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        ax = plot_iris_distribution(iris, index=0, show=False)
        assert ax is not None

    @staticmethod
    def test_not_fitted_error():
        """NotFittedError is raised when IRIS is not fitted."""
        from spectrochempy_iris import IRIS
        from spectrochempy_iris import plot_iris_distribution

        from spectrochempy.utils.exceptions import NotFittedError

        iris = IRIS()
        with pytest.raises(NotFittedError):
            plot_iris_distribution(iris, index=0, show=False)

    @staticmethod
    def test_default_index_all_lambdas():
        """Default index=None plots all lambdas, returns list."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        axes = plot_iris_distribution(iris, show=False)
        assert isinstance(axes, list)
        assert len(axes) == len(iris._lambdas)
        plt.close("all")

    @staticmethod
    def test_provided_ax_clear_false():
        """With clear=False and single index, existing artists are preserved."""
        import matplotlib.pyplot as plt
        from spectrochempy_iris import plot_iris_distribution

        iris = _fitted_iris()
        fig, ax = plt.subplots()
        plot_iris_distribution(iris, index=0, ax=ax, show=False)
        n_artists_before = len(ax.collections)

        plot_iris_distribution(iris, index=0, ax=ax, show=False, clear=False)
        n_artists_after = len(ax.collections)
        assert n_artists_after > n_artists_before
        plt.close()
