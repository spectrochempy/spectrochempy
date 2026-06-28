import numpy as np

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset


def _gaussian_profiles():
    time = scp.linspace(0.0, 1.0, 5)
    centers = (0.2, 0.5, 0.8)
    profiles = [np.exp(-((time - center) ** 2) / 0.05) for center in centers]
    return time, profiles


class TestWorkflowMathBaseline:
    """
    Characterize the current public behavior seen by workflow-style notebooks.

    This suite documents the exact gap observed during Workflow Assistant
    notebook generation:

    - unary NumPy ufuncs like ``np.exp`` dispatch to ``NDDataset``;
    - SpectroChemPy does not expose a matching top-level ``scp.exp`` alias;
    - a native ``scp.stack(..., axis=1)`` path now exists for 1D profiles;
    - ``scp.concatenate(..., axis=1)`` now provides the same narrow promotion
      behavior for 1D profiles.
    """

    def test_numpy_exp_dispatches_to_nddataset(self):
        time = scp.linspace(0.0, 1.0, 5)
        result = np.exp(-((time - 0.5) ** 2) / 0.05)

        assert isinstance(result, NDDataset)
        assert result.shape == (5,)
        assert result.dims == ["x"]
        np.testing.assert_allclose(
            result.data,
            np.exp(-((time.data - 0.5) ** 2) / 0.05),
        )

    def test_top_level_scp_exp_is_not_exposed(self):
        assert not hasattr(scp, "exp")

    def test_numpy_column_stack_returns_plain_ndarray(self):
        _, profiles = _gaussian_profiles()

        result = np.column_stack(profiles)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 3)

    def test_stack_of_1d_profiles_axis_1_returns_2d_dataset(self):
        _, profiles = _gaussian_profiles()

        result = scp.stack(profiles, axis=1)

        assert isinstance(result, NDDataset)
        assert result.shape == (5, 3)
        assert result.dims == ["x", "y"]
        assert result.y.labels is not None
        assert len(result.y.labels) == 3

    def test_concatenate_1d_profiles_axis_1_returns_2d_dataset(self):
        _, profiles = _gaussian_profiles()

        result = scp.concatenate(*profiles, axis=1)

        assert isinstance(result, NDDataset)
        assert result.shape == (5, 3)
        assert result.dims == ["x", "y"]
        assert result.y.labels is not None
        assert len(result.y.labels) == 3
