"""Retained spectral geometry must agree before automatic import assembly."""

from zipfile import ZipFile

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.readers.importer import merge_datasets


def spectrum(values=(4000.0, 3000.0, 2000.0), units="cm^-1", ndim=1):
    coord = scp.Coord(values, units=units)
    if ndim == 1:
        return scp.NDDataset([1, 2, 3], coordset=[coord])
    return scp.NDDataset([[1, 2, 3], [4, 5, 6]], coordset=[scp.Coord([10, 20]), coord])


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize(
    "change",
    ["shift", "reverse", "unit", "dimension", "missing", "labels", "nonfinite"],
)
def test_incompatible_retained_geometry_stays_separate(explicit, ndim, change):
    first = spectrum(ndim=ndim)
    second = first.copy()
    if change == "shift":
        second.x = scp.Coord([3900, 2900, 1900], units="cm^-1")
    elif change == "reverse":
        second = second[..., ::-1]
    elif change == "unit":
        second.x = scp.Coord([4000, 3000, 2000], units="s")
    elif change == "dimension":
        second.dims = ["z"] if ndim == 1 else ["z", "x"]
    elif change == "missing":
        second.x = scp.Coord()
    elif change == "labels":
        second.x.labels = ["a", "b", "c"]
    else:
        second.x = scp.Coord([4000, np.nan, 2000], units="cm^-1")
    kwargs = {"merge": True} if explicit else {}
    with pytest.warns(UserWarning, match="Imported datasets kept in separate groups"):
        result = merge_datasets([first, second], sortbydate=False, **kwargs)
    assert len(result) == 2
    assert result[0] is first and result[1] is second


@pytest.mark.parametrize("ndim", [1, 2])
def test_equivalent_units_keep_masks_values_and_inputs(ndim):
    first = spectrum(ndim=ndim)
    second = spectrum([400000, 300000, 200000], "m^-1", ndim=ndim)
    second[..., 1] = scp.MASKED
    before = [d.copy() for d in (first, second)]
    result = merge_datasets([first, second], merge=True, sortbydate=False)[0]
    assert result.shape == ((2, 3) if ndim == 1 else (4, 3))
    np.testing.assert_array_equal(result.x.data, first.x.data)
    assert result.x.units == first.x.units
    assert np.count_nonzero(result.mask) == ndim
    for original, snapshot in zip((first, second), before, strict=True):
        np.testing.assert_array_equal(original.data, snapshot.data)
        np.testing.assert_array_equal(original.mask, snapshot.mask)
        assert original.coordset == snapshot.coordset


def test_observation_coordinates_may_differ():
    first, second = spectrum(ndim=2), spectrum(ndim=2)
    second.y = scp.Coord([30, 40])
    result = merge_datasets([first, second], merge=True, sortbydate=False)[0]
    np.testing.assert_array_equal(result.y.data, [10, 20, 30, 40])


def test_compatible_subset_merges_without_losing_incompatible_input():
    first, second = spectrum(), spectrum()
    shifted = spectrum([3900, 2900, 1900])
    with pytest.warns(UserWarning, match="separate groups"):
        result = merge_datasets([first, shifted, second], merge=True, sortbydate=False)
    assert len(result) == 2
    assert result[0].shape == (2, 3)
    assert result[1] is shifted


@pytest.mark.parametrize("difference", ["values", "active", "labels"])
def test_all_alternative_coordinates_are_checked(difference):
    first = spectrum()
    first.set_coordset(
        x=[scp.Coord([4000, 3000, 2000], units="cm^-1"), scp.Coord([1, 2, 3])]
    )
    second = first.copy()
    if difference == "values":
        second.x.coords[1].data = [1, 2, 4]
    elif difference == "labels":
        second.x.coords[1].labels = ["a", "b", "c"]
    else:
        second.x.select(2)
    with pytest.warns(UserWarning, match="separate groups"):
        assert len(merge_datasets([first, second], merge=True)) == 2


@pytest.mark.parametrize("route", ["paths", "directory", "bytes", "zip"])
@pytest.mark.parametrize("shifted", [False, True])
def test_public_import_routes(tmp_path, route, shifted):
    first = spectrum()
    second = spectrum([3900, 2900, 1900]) if shifted else spectrum()
    paths = [first.dump(tmp_path / "first.scp"), second.dump(tmp_path / "second.scp")]
    if route == "directory":
        read = lambda: scp.read_dir(tmp_path, merge=True, sortbydate=False)
    elif route == "bytes":
        read = lambda: scp.read(
            {p.name: p.read_bytes() for p in paths}, merge=True, sortbydate=False
        )
    elif route == "zip":
        archive = tmp_path / "spectra.zip"
        with ZipFile(archive, "w") as bundle:
            for path in paths:
                bundle.write(path, arcname=path.name)
        read = lambda: scp.read(archive, merge=True, sortbydate=False)
    else:
        read = lambda: scp.read(paths, merge=True, sortbydate=False)
    if shifted:
        with pytest.warns(UserWarning, match="separate groups"):
            result = read()
        assert len(result) == 2
        assert {tuple(d.x.data) for d in result} == {
            (4000, 3000, 2000),
            (3900, 2900, 1900),
        }
    else:
        result = read()
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (2, 3)


def test_merge_false_does_not_validate_or_change_inputs():
    inputs = [spectrum(), spectrum([3900, 2900, 1900])]
    assert merge_datasets(inputs, merge=False) is inputs


@pytest.mark.parametrize("shift, compatible", [(1e-12, True), (1e-8, False)])
def test_only_numeric_roundoff_is_tolerated(shift, compatible):
    first = spectrum()
    first.x = scp.Coord([1000, 1234, 1800], rounding=False, sigdigits=16)
    second = first.copy()
    second.x = scp.Coord([1000 + shift, 1234, 1800], rounding=False, sigdigits=16)
    if compatible:
        assert len(merge_datasets([first, second], merge=True, sortbydate=False)) == 1
    else:
        with pytest.warns(UserWarning, match="separate groups"):
            assert len(merge_datasets([first, second], merge=True)) == 2


def test_label_only_coordinates_must_agree():
    first, second = spectrum(), spectrum()
    first.x = scp.Coord(labels=["one", "two", "three"])
    second.x = scp.Coord(labels=["one", "two", "other"])
    with pytest.warns(UserWarning, match="separate groups"):
        assert len(merge_datasets([first, second], merge=True)) == 2


def test_all_retained_dimensions_are_checked_in_3d():
    first = scp.NDDataset(
        np.ones((2, 3, 4)),
        coordset=[scp.Coord([1, 2]), scp.Coord([1, 2, 3]), scp.Coord([1, 2, 3, 4])],
    )
    second = first.copy()
    second.y = scp.Coord([1, 2, 4])
    with pytest.warns(UserWarning, match="retained dimension 'y'"):
        assert len(merge_datasets([first, second], merge=True)) == 2


def test_intensity_unit_conversion_and_complex_promotion_remain_supported():
    first, second = spectrum(), spectrum()
    first.units, second.units = "m", "cm"
    second.data = np.array([100 + 10j, 200 + 20j, 300 + 30j])
    result = merge_datasets([first, second], merge=True, sortbydate=False)[0]
    assert result.units == first.units
    np.testing.assert_allclose(result.data[1], [1 + 0.1j, 2 + 0.2j, 3 + 0.3j])


def test_different_origins_cannot_bypass_geometry_check():
    first, second = spectrum(), spectrum([3900, 2900, 1900])
    first.origin, second.origin = "ir", "raman"
    with pytest.warns(UserWarning, match="separate groups"):
        assert len(merge_datasets([first, second], merge=True)) == 2


@pytest.mark.parametrize("sortbydate", [None, False, True])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("partition", ["geometry", "quantity", "origin"])
def test_sort_option_applies_to_every_compatible_group(
    sortbydate, complex_data, partition
):
    inputs = []
    for index, timestamp in enumerate([40, 30, 20, 10]):
        values = np.array([[index + 1, index + 11, index + 21]], dtype=float)
        if complex_data:
            values = values + 1j * (values + 100)
        dataset = scp.NDDataset(values)
        dataset.set_coordset(
            y=scp.Coord([timestamp], labels=[f"source {index}"]),
            x=scp.Coord([1, 2, 4 if partition == "geometry" and index % 2 else 3]),
        )
        dataset.origin = (
            f"source {index % 2}" if partition == "origin" else "same source"
        )
        if partition == "quantity":
            dataset.units = "transmittance" if index % 2 else "absorbance"
        dataset[0, index % 3] = scp.MASKED
        inputs.append(dataset)
    before = [dataset.copy() for dataset in inputs]
    options = {} if sortbydate is None else {"sortbydate": sortbydate}
    if partition == "origin":
        result = merge_datasets(inputs, **options)
    else:
        with pytest.warns(UserWarning, match="kept in separate groups"):
            result = merge_datasets(inputs, merge=True, **options)
    assert len(result) == 2
    for group, indices in zip(result, [[0, 2], [1, 3]], strict=True):
        if sortbydate is not False:
            indices = list(reversed(indices))
        np.testing.assert_array_equal(
            group.data, np.concatenate([inputs[index].data for index in indices])
        )
        np.testing.assert_array_equal(
            group.mask, np.concatenate([inputs[index].mask for index in indices])
        )
        np.testing.assert_array_equal(
            group.y.data, [inputs[i].y.data[0] for i in indices]
        )
        assert group.y.labels.tolist() == [f"source {i}" for i in indices]
        assert np.iscomplexobj(group.data) is complex_data
    for dataset, snapshot in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(dataset.data, snapshot.data)
        np.testing.assert_array_equal(dataset.mask, snapshot.mask)
        assert dataset.coordset == snapshot.coordset
