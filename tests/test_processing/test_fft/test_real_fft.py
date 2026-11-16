# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

import spectrochempy as scp


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
def test_real_fft_roundtrip_along_named_axis(size, norm):
    values = np.arange(3.0 * size).reshape(size, 3)
    data = scp.NDDataset(values, units="absorbance")
    data.y = scp.Coord(3 + np.arange(size) * 2, title="time", units="ms")
    data.x = scp.Coord([10, 20, 30], title="energy", units="eV")

    result = scp.rfft(data, dim="y", norm=norm)

    assert result.x == data.x
    assert result.units == data.units
    assert result.y.units.is_compatible_with("Hz")
    np.testing.assert_allclose(
        result.y.to("Hz").data, np.fft.rfftfreq(size, 0.002), rtol=1e-3
    )
    np.testing.assert_allclose(result.data, np.fft.rfft(values, axis=0, norm=norm))

    restored = result.irfft(dim="y")
    assert restored.y.title == "time"
    np.testing.assert_allclose(restored.y.data, data.y.data)
    np.testing.assert_allclose(restored.data, values, atol=1e-13)


def test_real_fft_rejects_split_axes_whose_mean_hides_irregular_sampling():
    data = scp.NDDataset(np.arange(6.0))
    data.set_coordset(x=scp.Coord([0, 1.2, 2, 3, 3.8, 5], units="s"))
    cycles = data.reshape((2, 3), coord_policy="split")
    np.testing.assert_allclose(cycles.x.data, [0, 1, 2])

    with pytest.raises(ValueError, match="Within-block"):
        cycles.rfft()

    cycles.x = scp.Coord([0, 1, 2], units="s")
    np.testing.assert_allclose(cycles.rfft().data, np.fft.rfft(cycles.data))


def test_irfft_converts_origin_when_frequency_unit_changes():
    data = scp.NDDataset(np.arange(8.0))
    data.x = scp.Coord(3 + 2 * np.arange(8), units="ms")
    spectrum = data.rfft()
    spectrum.x.ito("Hz")

    restored = spectrum.irfft()
    assert restored.x.units.is_compatible_with("s")
    np.testing.assert_allclose(restored.x.data, data.x.data * 0.001)


@pytest.mark.parametrize("operation", [scp.rfft, scp.irfft])
def test_real_fft_rejects_empty_data(operation):
    with pytest.raises(ValueError, match="nonempty"):
        operation(scp.NDDataset())


def test_coordinate_display_rounding_does_not_hide_irregular_sampling():
    data = scp.NDDataset(np.arange(6.0))
    data.set_coordset(
        x=scp.Coord(
            10000 + np.array([0, 1.0004, 2, 3, 3.9996, 5]),
            sigdigits=15,
            linearize_below=0,
        )
    )
    data.x.sigdigits = 4

    with pytest.raises(ValueError, match="uniformly spaced"):
        data.rfft()
    with pytest.raises(ValueError, match="Within-block"):
        data.reshape((2, 3), coord_policy="split").rfft()


def test_resplitting_retains_irregular_sampling_provenance():
    data = scp.NDDataset(np.arange(8.0))
    data.set_coordset(x=scp.Coord([0, 1.2, 2, 3, 4, 4.8, 6, 7]))
    cycles = data.reshape((2, 4), coord_policy="split")
    split_again = cycles.reshape((2, 2, 2), dims=("z", "y", "x"), coord_policy="split")

    for dim in ("y", "x"):
        with pytest.raises(ValueError, match="Within-block"):
            split_again.rfft(dim=dim)


def test_real_fft_handles_extreme_spacing_without_invalid_coordinates():
    data = scp.NDDataset([1.0, 2.0])
    data.set_coordset(x=scp.Coord([0.0, 1e308], rounding=False))
    spectrum = data.rfft()
    assert spectrum.x.data[1] == pytest.approx(5e-309, abs=0.0)
    np.testing.assert_allclose(spectrum.irfft().x.data, data.x.data)

    data.set_coordset(x=scp.Coord([0.0, 1e-310], rounding=False))
    with pytest.raises(ValueError, match="frequency grid cannot be represented"):
        data.rfft()


def test_real_fft_preserves_coordinate_referenced_through_an_auxiliary_name():
    data = scp.NDDataset(np.arange(24.0).reshape(4, 6))
    data.set_coordset(
        y=scp.Coord(np.arange(4)),
        z=scp.Coord(np.arange(6) + 10, units="eV"),
        x="z",
    )

    result = data.rfft(dim="y")
    assert result.x.units == data.coordset["z"].units
    np.testing.assert_array_equal(result.x.data, data.coordset["z"].data)


def test_real_fft_rejects_unrepresentable_results():
    data = scp.NDDataset([1e308, 1e308])
    with pytest.raises(ValueError, match="Fourier coefficients cannot be represented"):
        data.rfft()
    with pytest.raises(ValueError, match="reconstructed samples cannot be represented"):
        data.irfft(norm="forward")
