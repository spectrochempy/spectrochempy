import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_histogram_counts_unmasked_finite_values_across_dimensions():
    dataset = scp.NDDataset(
        [[0.25, 0.75, 1.75], [0.5, np.nan, np.inf]],
        mask=[[False, False, False], [True, False, False]],
        units="MHz",
        title="Coupling",
    )

    ax = dataset.plot(method="hist", bins=[0, 1, 2], show=False)

    assert [patch.get_height() for patch in ax.patches] == [2, 1]
    assert "Coupling" in ax.get_xlabel()
    assert "MHz" in ax.get_xlabel()
    assert ax.get_ylabel() == "Count"


def test_histogram_density_integrates_to_one_with_unequal_bins():
    dataset = scp.NDDataset([0.5, 0.75, 2.0, 3.0], units="MHz")

    ax = dataset.plot(method="hist", bins=[0, 1, 3], density=True, show=False)

    heights = [patch.get_height() for patch in ax.patches]
    widths = [patch.get_width() for patch in ax.patches]
    assert heights == pytest.approx([0.5, 0.25])
    assert np.dot(heights, widths) == pytest.approx(1)
    assert "Probability density" in ax.get_ylabel()
    assert r"\mathrm{MHz}^{-1}" in ax.get_ylabel()


def test_histogram_overlays_outlines_and_exports_figure(tmp_path):
    dataset = scp.NDDataset([0.25, 1.25])
    target = tmp_path / "histogram.png"
    _, ax = plt.subplots()

    first = dataset.plot(method="hist", ax=ax, histtype="step", show=False)
    second = dataset.plot(
        method="hist", ax=ax, histtype="step", output=target, show=False
    )

    assert first is second is ax
    assert len(ax.patches) == 2
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_histogram_can_reuse_current_axes_without_clear():
    _, ax = plt.subplots()
    dataset = scp.NDDataset([0.25, 1.25])

    assert dataset.plot(method="hist", clear=False, show=False) is ax


def test_empty_histogram_counts_and_density():
    dataset = scp.NDDataset([1.0, 2.0], mask=[True, True])
    ax = dataset.plot(method="hist", bins=2, show=False)

    assert [patch.get_height() for patch in ax.patches] == [0, 0]
    with pytest.raises(ValueError, match="nonzero finite mass"):
        dataset.plot(method="hist", density=True, show=False)


def test_histogram_rejects_complex_data():
    dataset = scp.NDDataset([1 + 2j, 3 + 4j])

    with pytest.raises(ValueError, match="require real values"):
        dataset.plot(method="hist", show=False)


def test_histogram_filters_weights_with_their_samples():
    dataset = scp.NDDataset([0.25, 0.75, 1.25], mask=[False, True, False])

    ax = dataset.plot(method="hist", bins=[0, 1, 2], weights=[2, 100, 3], show=False)

    assert [patch.get_height() for patch in ax.patches] == [2, 3]


def test_horizontal_histogram_places_density_on_x_axis():
    dataset = scp.NDDataset([0.5, 0.75, 2], title="Coupling", units="MHz")

    ax = dataset.plot(
        method="hist",
        bins=[0, 1, 3],
        density=True,
        orientation="horizontal",
        show=False,
    )

    assert [patch.get_width() for patch in ax.patches] == pytest.approx([2 / 3, 1 / 6])
    assert r"\mathrm{MHz}^{-1}" in ax.get_xlabel()
    assert "Coupling" in ax.get_ylabel()
    assert "MHz" in ax.get_ylabel()
