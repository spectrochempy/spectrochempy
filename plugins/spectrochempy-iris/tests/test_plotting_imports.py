# ruff: noqa: S101, PLC0415  # assert/local imports allowed in plugin tests
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""IRIS plugin integration checks for exported plotting helpers."""

import pytest


@pytest.mark.plugin
class TestIrisImports:
    """IRIS plugin integration checks for exported plotting helpers."""

    @staticmethod
    def _check_plugin():
        pytest.importorskip(
            "spectrochempy_iris",
            reason="requires the optional spectrochempy-iris plugin",
        )

    @pytest.mark.plugin
    def test_plot_iris_lcurve_import(self):
        """Test import of plot_iris_lcurve."""
        self._check_plugin()
        from spectrochempy_iris import plot_iris_lcurve

        assert callable(plot_iris_lcurve)

    @pytest.mark.plugin
    def test_plot_iris_distribution_import(self):
        """Test import of plot_iris_distribution."""
        self._check_plugin()
        from spectrochempy_iris import plot_iris_distribution

        assert callable(plot_iris_distribution)

    @pytest.mark.plugin
    def test_plot_iris_merit_import(self):
        """Test import of plot_iris_merit."""
        self._check_plugin()
        from spectrochempy_iris import plot_iris_merit

        assert callable(plot_iris_merit)
