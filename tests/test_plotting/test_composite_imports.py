# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Minimal tests for composite plotting modules to improve coverage.

These tests ensure the modules can be imported and basic functions are callable.
"""

import numpy as np
import pytest


class TestPlotMeritImports:
    """Test that plotmerit module can be imported and has expected functions."""

    def test_plot_merit_import(self):
        """Test import of plot_merit."""
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        assert callable(plot_merit)

    def test_plot_compare_import(self):
        """Test import of plot_compare."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        assert callable(plot_compare)

    def test_plot_compare_makes_reconstructed_trace_visible(self, sample_1d_dataset):
        """The reconstructed profile should remain visually distinct."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, show=False)

        assert len(ax.lines) == 3
        assert ax.lines[1].get_color() == "tab:blue"
        assert ax.lines[2].get_color() == "tab:orange"
        assert ax.lines[2].get_linestyle() == "--"

    def test_plot_compare_scatter_mode_uses_markers(self, sample_1d_dataset):
        """Scatter mode should render point markers rather than solid lines."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, kind="scatter", show=False)

        assert len(ax.lines) == 3
        for line in ax.lines:
            assert line.get_marker() == "o"
            assert line.get_linestyle() == "None"

    def test_plot_compare_nb_traces_limits_display(self, sample_2d_dataset):
        """nb_traces should reduce the number of rendered traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_2d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, nb_traces=3, show=False)

        assert len(ax.lines) == 9

    def test_plot_compare_offset_moves_residual_down(self, sample_1d_dataset):
        """Offset should separate the residual from the main traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X * 0.9
        X_ref.name = "fit"

        ax = plot_compare(X, X_ref, offset=20, show=False)

        signal_range = max(np.nanmax(X.data), np.nanmax(X_ref.data)) - min(
            np.nanmin(X.data),
            np.nanmin(X_ref.data),
        )
        expected_shift = 0.2 * signal_range
        expected_residual = (X - X_ref).data - expected_shift

        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_residual)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), X.data)
        np.testing.assert_allclose(ax.lines[2].get_ydata(), X_ref.data)

    def test_plot_compare_uses_scpy_figure_size_preference(self, sample_1d_dataset):
        """Standalone comparison figures should follow the current SCPy figure size."""
        from spectrochempy import preferences
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        original_figsize = tuple(preferences.figure.figsize)
        preferences.figure.figsize = (7.0, 3.0)
        try:
            ax = plot_compare(X, X_ref, show=False)
            width, height = ax.figure.get_size_inches()
            np.testing.assert_allclose((width, height), (7.0, 3.0))
            assert ax.figure.get_tight_layout() is True
        finally:
            preferences.figure.figsize = original_figsize

    def test_plot_compare_accepts_short_legend_labels_and_position(
        self,
        sample_1d_dataset,
    ):
        """Legend labels and location should be customizable for compact figures."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        X_ref.name = "fit"

        ax = plot_compare(
            X,
            X_ref,
            exp_label="exp",
            calc_label="fit",
            resid_label="res",
            legend_loc="upper left",
            show=False,
        )

        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["exp", "fit", "res"]

    def test_plot_compare_legend_uses_the_plotted_lines(self):
        """The legend refers to labelled plotted lines, with readable defaults."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(3, 20))
        X_ref = X * 0.9

        _, ax = plt.subplots()
        (component,) = ax.plot([0, 1], [0, 1], label="component")
        ax = plot_compare(X, X_ref, ax=ax, clear=False, show=False)

        legend = ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "experimental",
            "calculated",
            "difference",
            "component",
        ]
        handles, _ = ax.get_legend_handles_labels()
        assert component in handles
        assert all(handle in ax.lines for handle in handles)
        assert [line.get_label() for line in ax.lines[1:]] == [
            "difference",
            "_nolegend_",
            "_nolegend_",
            "experimental",
            "_nolegend_",
            "_nolegend_",
            "calculated",
            "_nolegend_",
            "_nolegend_",
        ]
        styles = [
            (handle.get_color(), handle.get_linestyle())
            for handle in legend.legend_handles
        ]
        assert styles[:3] == [("tab:blue", "-"), ("tab:orange", "--"), ("0.6", "-")]
        plt.close("all")

    def test_plot_compare_default_labels_keep_names_set_on_datasets(self):
        """Names set on the datasets are kept; automatic names are replaced."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(3, 20))
        X_ref = X * 0.9

        def legend_texts(ax):
            return [text.get_text() for text in ax.get_legend().get_texts()]

        X_ref.name = f"{X.name}_MCRALS.reconstruction"
        ax = plot_compare(X, X_ref, residual=False, show=False)
        assert legend_texts(ax) == ["experimental", "calculated"]

        X.name = "measured"
        X_ref.name = "measured_MCRALS.reconstruction"
        ax = plot_compare(X, X_ref, residual=False, show=False)
        assert legend_texts(ax) == ["measured", "measured_MCRALS.reconstruction"]
        plt.close("all")

    @pytest.mark.parametrize(
        ("exp_name", "calc_name"),
        [
            ("IR_20240315", "fit_ab12cd34"),
            ("sample_deadbeef", "Coordinates_12345678_fit"),
        ],
    )
    def test_plot_compare_default_labels_keep_user_names_resembling_ids(
        self, exp_name, calc_name
    ):
        """User names that look like automatic ids remain the default labels."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(3, 20), name=exp_name)
        X_ref = scp.NDDataset(X.data * 0.9, name=calc_name)

        ax = plot_compare(X, X_ref, residual=False, show=False)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            exp_name,
            calc_name,
        ]
        plt.close("all")

    @pytest.mark.parametrize(
        ("exp_name", "calc_name"),
        [
            ("NDDataset_1a2b3c4d", "NDDataset_1a2b3c4d_MCRALS.reconstruction"),
            ("NDComplexArray_0f0f0f0f", "Coord_12345678_PCA.reconstruction"),
            ("standardized NDDataset_1a2b3c4d", "scaled NDDataset_1a2b3c4d"),
            ("standardized standardized NDDataset_1a2b3c4d", "NDDataset_1a2b3c4d"),
        ],
    )
    def test_plot_compare_default_labels_replace_automatic_names(
        self, exp_name, calc_name
    ):
        """Automatic ids, also as analysis name prefixes, fall back to roles."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(3, 20), name=exp_name)
        X_ref = scp.NDDataset(X.data * 0.9, name=calc_name)

        ax = plot_compare(X, X_ref, residual=False, show=False)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "experimental",
            "calculated",
        ]
        plt.close("all")

    def test_plot_compare_labels_keyword_is_deprecated_and_ignored(self):
        """``labels`` warns and leaves the per-role labels in charge."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(2, 20))
        X_ref = X * 0.9

        with pytest.warns(
            DeprecationWarning,
            match="`labels` argument of `plot_compare` is deprecated and ignored"
            ".*Use `exp_label`, `calc_label` and `resid_label` instead",
        ):
            ax = plot_compare(X, X_ref, labels=["a", "b"], exp_label="exp", show=False)

        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "exp",
            "calculated",
            "difference",
        ]
        assert {line.get_label() for line in ax.lines} == {
            "exp",
            "calculated",
            "difference",
            "_nolegend_",
        }
        plt.close("all")

    def test_plot_compare_legend_skips_hidden_and_repeated_entries(self):
        """No empty legend box, and repeated calls do not repeat entries."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(2, 20))
        X_ref = X * 0.9

        ax = plot_compare(
            X,
            X_ref,
            residual=False,
            exp_label="_hidden",
            calc_label="_hidden",
            show=False,
        )
        assert ax.get_legend() is None

        _, ax = plt.subplots()
        plot_compare(X, X_ref, ax=ax, clear=False, show=False)
        plot_compare(X, X_ref * 0.9, ax=ax, clear=False, show=False)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "experimental",
            "calculated",
            "difference",
        ]
        plt.close("all")

    def test_plot_merit_overlay_keeps_one_legend_entry_per_role(self):
        """Overlaid reconstructions do not repeat the comparison legend entries."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        rng = np.random.RandomState(42)
        X = scp.NDDataset(rng.randn(10, 8))
        X_hat = scp.NDDataset(rng.randn(3, 10, 8))
        pca = scp.PCA(n_components=2).fit(X)

        ax = plot_merit(pca, X=X, X_hat=X_hat, show=False)

        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "experimental",
            "calculated",
            "difference",
        ]
        plt.close("all")

    def test_plot_merit_standardized_pca_uses_role_labels(self):
        """PCA preprocessing prefixes on an automatic name keep role labels."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp

        X = scp.NDDataset(np.random.RandomState(0).rand(10, 8))
        pca = scp.PCA(n_components=2, standardized=True).fit(X)
        assert pca.X.name.startswith("standardized ")

        ax = pca.plot_merit(show=False)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "experimental",
            "calculated",
            "difference",
        ]
        plt.close("all")

    def test_plot_compare_legend_keeps_both_datasets_with_the_same_name(self):
        """A smoothed copy keeps its name, and both lines stay in the legend."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(
            np.random.RandomState(0).rand(50),
            coordset=[scp.Coord(np.linspace(1000.0, 2000.0, 50))],
            name="noisy spectrum",
        )
        smoothed = X.smooth(size=7)
        assert smoothed.name == X.name

        ax = plot_compare(X, smoothed, show=False)
        legend = ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "noisy spectrum",
            "noisy spectrum",
            "difference",
        ]
        assert [handle.get_linestyle() for handle in legend.legend_handles] == [
            "-",
            "--",
            "-",
        ]
        plt.close("all")

    def test_plot_compare_legend_keeps_explicit_equal_labels(self):
        """Equal labels of one call are kept; other artists are deduplicated."""
        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(2, 20))
        X_ref = X * 0.9

        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="same")
        (other,) = ax.plot([0, 1], [1, 0], label="other")
        ax = plot_compare(
            X,
            X_ref,
            ax=ax,
            clear=False,
            exp_label="same",
            calc_label="same",
            show=False,
        )

        legend = ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "same",
            "same",
            "difference",
            "other",
        ]
        assert legend.legend_handles[3].get_color() == other.get_color()
        plt.close("all")

    def test_plot_compare_labels_deprecation_points_at_the_caller(self):
        """The ``labels`` warning is attributed to the user's call."""
        import warnings

        import matplotlib.pyplot as plt

        import spectrochempy as scp
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = scp.NDDataset(np.random.RandomState(0).rand(10, 8))
        pca = scp.PCA(n_components=2).fit(X)

        calls = [
            lambda: plot_compare(X, X * 0.9, labels=["a", "b"], show=False),
            lambda: X.plot_compare(X * 0.9, labels=["a", "b"], show=False),
            lambda: pca.plot_merit(labels=["a", "b"], show=False),
        ]
        for call in calls:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                call()
            deprecations = [
                record
                for record in records
                if issubclass(record.category, DeprecationWarning)
            ]
            assert len(deprecations) == 1
            assert deprecations[0].filename == __file__
        plt.close("all")

    def test_plot_compare_color_kwarg(self, sample_1d_dataset):
        """'color' in kwargs should distribute to all three traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        ax = plot_compare(X, X_ref, color="red", show=False)
        assert len(ax.lines) == 3
        assert ax.lines[0].get_color() == "red"
        assert ax.lines[1].get_color() == "red"
        assert ax.lines[2].get_color() == "red"

    def test_plot_compare_linestyle_kwarg(self, sample_1d_dataset):
        """'linestyle' in kwargs should apply to all three traces."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        ax = plot_compare(X, X_ref, linestyle=":", show=False)
        assert len(ax.lines) == 3
        assert ax.lines[0].get_linestyle() == ":"
        assert ax.lines[1].get_linestyle() == ":"
        assert ax.lines[2].get_linestyle() == ":"

    def test_plot_compare_per_category_take_precedence_over_kwargs(
        self, sample_1d_dataset
    ):
        """Explicit per-category color params should take precedence over kwargs color."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        ax = plot_compare(
            X, X_ref, exp_c="blue", calc_c="green", color="red", show=False
        )
        assert len(ax.lines) == 3
        assert ax.lines[1].get_color() == "blue"
        assert ax.lines[2].get_color() == "green"

    def test_plot_compare_c_alias_from_kwargs(self, sample_1d_dataset):
        """'c' alias in kwargs should be treated as color."""
        from spectrochempy.plotting.composite.plotmerit import plot_compare

        X = sample_1d_dataset
        X_ref = X.copy()
        ax = plot_compare(X, X_ref, c="purple", show=False)
        assert len(ax.lines) == 3
        assert ax.lines[0].get_color() == "purple"
        assert ax.lines[1].get_color() == "purple"
        assert ax.lines[2].get_color() == "purple"


class TestMultiplotImports:
    """Test that multiplot module can be imported."""

    def test_multiplot_import(self):
        """Test import of multiplot."""
        from spectrochempy.plotting.multiplot import multiplot

        assert callable(multiplot)

    def test_multiplot_scatter_import(self):
        """Test import of multiplot_scatter."""
        from spectrochempy.plotting.multiplot import multiplot_scatter

        assert callable(multiplot_scatter)

    def test_multiplot_lines_import(self):
        """Test import of multiplot_lines."""
        from spectrochempy.plotting.multiplot import multiplot_lines

        assert callable(multiplot_lines)


class TestProfileImports:
    """Test that profile module can be imported."""

    def test_profile_manager_import(self):
        """Test import of PlotProfileManager."""
        from spectrochempy.plotting.profile import PlotProfileManager

        assert PlotProfileManager is not None

    def test_get_plot_profile_import(self):
        """Test import of get_plot_profile."""
        from spectrochempy.plotting.profile import get_plot_profile

        assert callable(get_plot_profile)

    def test_set_plot_profile_import(self):
        """Test import of set_plot_profile."""
        from spectrochempy.plotting.profile import set_plot_profile

        assert callable(set_plot_profile)
