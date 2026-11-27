"""Known signal quantities and declared modalities constrain import merging."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.readers.importer import merge_datasets


def spectrum(units=None, **kwargs):
    return scp.NDDataset(
        [0.1, 0.2, 0.3],
        units=units,
        coordset=[scp.Coord([4000, 3000, 2000], units="cm^-1")],
        **kwargs,
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "units",
    [
        ("absorbance", "transmittance"),
        ("absolute_transmittance", "absorbance"),
        ("absorbance", "Kubelka_Munk"),
        ("transmittance", "Kubelka_Munk"),
        ("milliabsorbance", "transmittance"),
        ("absorbance", "millitransmittance"),
    ],
)
def test_known_quantity_conflicts_stay_separate(units, explicit):
    first, second = map(spectrum, units)
    before = second.data.copy()
    with pytest.warns(UserWarning, match="signal quantities differ"):
        result = merge_datasets(
            [first, second], sortbydate=False, **({"merge": True} if explicit else {})
        )
    assert len(result) == 2
    assert result[0] is first and result[1] is second
    np.testing.assert_array_equal(second.data, before)


@pytest.mark.parametrize(
    "first_unit, second_unit, scale",
    [
        ("transmittance", "absolute_transmittance", 100),
        ("absorbance", "milliabsorbance", 0.001),
        ("m", "cm", 0.01),
    ],
)
def test_compatible_unit_scales_still_merge(first_unit, second_unit, scale):
    first, second = spectrum(first_unit), spectrum(second_unit)
    result = merge_datasets([first, second], merge=True, sortbydate=False)[0]
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result.data[1], second.data * scale)
    assert result.units == first.units


@pytest.mark.parametrize("declaration", ["RAMAN SPECTRUM", "NMR SPECTRUM"])
def test_explicit_modality_conflicts_stay_separate(declaration):
    first, second = spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    second.meta.jcamp_data_type = declaration
    with pytest.warns(UserWarning, match="declared JCAMP modalities differ"):
        assert len(merge_datasets([first, second], merge=True)) == 2


@pytest.mark.parametrize("unknown", [None, "custom data", ["RAMAN SPECTRUM"]])
def test_unknown_metadata_is_not_assigned_a_modality(unknown):
    first, second = spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    second.meta.jcamp_data_type = unknown
    assert len(merge_datasets([first, second], merge=True, sortbydate=False)) == 1


def test_free_titles_and_vendor_origins_do_not_define_identity():
    first = spectrum("absorbance", title="my spectrum", origin="omnic")
    second = spectrum("absorbance", title="different title", origin="opus-AB")
    assert len(merge_datasets([first, second], merge=True, sortbydate=False)) == 1


def test_unknown_input_cannot_bridge_conflicting_known_quantities():
    inputs = [spectrum(), spectrum("absorbance"), spectrum("transmittance")]
    with pytest.warns(UserWarning, match="signal quantities differ"):
        result = merge_datasets(inputs, merge=True, sortbydate=False)
    assert len(result) >= 2
    assert sum(d.shape[0] if d.ndim == 2 else 1 for d in result) == 3


@pytest.mark.parametrize("route", ["native", "jcamp"])
def test_public_import_preserves_conflicting_quantities(tmp_path, route):
    first, second = spectrum("absorbance"), spectrum("transmittance")
    if route == "native":
        paths = [
            first.dump(tmp_path / "first.scp"),
            second.dump(tmp_path / "second.scp"),
        ]
    else:
        first, second = (
            scp.NDDataset(
                [dataset.data],
                units=dataset.units,
                coordset=[scp.Coord([0]), dataset.x.copy()],
            )
            for dataset in (first, second)
        )
        paths = [
            first.write_jcamp(tmp_path / "first.jdx", confirm=False),
            second.write_jcamp(tmp_path / "second.jdx", confirm=False),
        ]
    with pytest.warns(UserWarning, match="signal quantities differ"):
        result = scp.read(paths, merge=True, sortbydate=False)
    assert len(result) == 2
    assert {d.units for d in result} == {first.units, second.units}


def test_merge_false_preserves_inputs_with_conflicting_identity():
    inputs = [spectrum("absorbance"), spectrum("transmittance")]
    assert merge_datasets(inputs, merge=False) is inputs


def test_existing_uvvis_grid_restriction_is_preserved():
    first, second = spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    second.meta.jcamp_data_type = "UV/VIS SPECTRUM"
    result = merge_datasets([first, second], merge=True)
    assert len(result) == 2


def test_known_modality_survives_an_intermediate_import_assembly():
    first, unknown, conflict = spectrum(), spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    conflict.meta.jcamp_data_type = "RAMAN SPECTRUM"
    intermediate = merge_datasets([unknown, first], merge=True, sortbydate=False)[0]
    other = merge_datasets([conflict, conflict.copy()], merge=True, sortbydate=False)[0]
    with pytest.warns(UserWarning, match="declared JCAMP modalities differ"):
        assert len(merge_datasets([intermediate, other], merge=True)) == 2


def test_known_quantity_survives_an_intermediate_unitless_assembly():
    first, unknown = spectrum("absorbance"), spectrum("dimensionless")
    conflict = spectrum("transmittance")
    intermediate = merge_datasets([unknown, first], merge=True, sortbydate=False)[0]
    other = merge_datasets([conflict, conflict.copy()], merge=True, sortbydate=False)[0]
    with pytest.warns(UserWarning, match="signal quantities differ"):
        assert len(merge_datasets([intermediate, other], merge=True)) == 2


def test_public_outputs_do_not_carry_temporary_import_constraints(tmp_path):
    first, second = spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    paths = [first.dump(tmp_path / "first.scp"), second.dump(tmp_path / "second.scp")]
    result = scp.read(paths, merge=True, sortbydate=False)
    assert result.shape == (2, 3)
    assert "_import_measurement_identity" not in result.__dict__


def test_explicit_identity_update_overrides_an_internal_carried_constraint():
    first, unknown = spectrum(), spectrum()
    first.meta.jcamp_data_type = "INFRARED SPECTRUM"
    result = merge_datasets([unknown, first], merge=True, sortbydate=False)[0]
    result.meta.jcamp_data_type = "RAMAN SPECTRUM"
    other = result.copy()
    assert len(merge_datasets([result, other], merge=True, sortbydate=False)) == 1


def test_explicit_jcamp_nmr_fid_and_spectrum_are_distinct():
    first, second = spectrum(), spectrum()
    first.meta.jcamp_data_type = "NMR FID"
    second.meta.jcamp_data_type = "NMR SPECTRUM"
    with pytest.warns(UserWarning, match="declared JCAMP modalities differ"):
        assert len(merge_datasets([first, second], merge=True)) == 2


@pytest.mark.parametrize("nested", [False, True])
def test_directory_assembly_cannot_erase_known_modality_constraints(tmp_path, nested):
    directories, paths = [], []
    for name, declarations in (
        ("a", (None, "INFRARED SPECTRUM")),
        ("b", ("RAMAN SPECTRUM", "RAMAN SPECTRUM")),
    ):
        directory = tmp_path / name
        directory.mkdir()
        directories.append(directory)
        for index, declaration in enumerate(declarations):
            dataset = spectrum()
            if declaration is not None:
                dataset.meta.jcamp_data_type = declaration
            paths.append(dataset.dump(directory / f"{index}.scp"))
    with pytest.warns(UserWarning, match="declared JCAMP modalities differ"):
        result = scp.read(
            directories if nested else paths, merge=True, sortbydate=False
        )
    assert len(result) == 2
    assert all(dataset.shape == (2, 3) for dataset in result)
    assert all(
        "_import_measurement_identity" not in dataset.__dict__ for dataset in result
    )


def test_failed_import_does_not_leak_the_nested_session(tmp_path):
    with pytest.raises(FileNotFoundError):
        scp.read(tmp_path / "missing.scp", local_only=True)
    paths = [spectrum().dump(tmp_path / f"{index}.scp") for index in range(2)]
    result = scp.read(paths, merge=True, sortbydate=False)
    assert result.shape == (2, 3)
    assert "_import_measurement_identity" not in result.__dict__


@pytest.fixture
def identity_provider(monkeypatch):
    from spectrochempy.plugins import manager

    registry = manager.plugin_manager.registry
    original = registry.get_handler

    def install(provider):
        monkeypatch.setattr(
            registry,
            "get_handler",
            lambda name: provider
            if name == "importer.measurement_identity"
            else original(name),
        )

    return install


def test_plugin_constraints_survive_intermediate_assembly(identity_provider):
    calls = []

    def provider(dataset):
        calls.append(dataset)
        nucleus = dataset.meta.get("test_nucleus")
        return {"constraints": {"test.nucleus": nucleus}} if nucleus else None

    identity_provider(provider)
    unknown, proton, carbon = spectrum(), spectrum(), spectrum()
    proton.meta.test_nucleus = "1H"
    carbon.meta.test_nucleus = "13C"
    series = merge_datasets([unknown, proton], merge=True, sortbydate=False)[0]
    assert len(calls) == 2
    other = merge_datasets([carbon, carbon.copy()], merge=True, sortbydate=False)[0]
    with pytest.warns(UserWarning, match="test.nucleus differ"):
        assert len(merge_datasets([series, other], merge=True)) == 2


@pytest.mark.parametrize("merge", [None, True])
def test_plugin_can_refuse_automatic_assembly(identity_provider, merge):
    identity_provider(lambda ds: {"constraints": {}, "reason": "needs a series axis"})
    first, second = spectrum(), spectrum()
    with pytest.warns(UserWarning, match="needs a series axis"):
        result = merge_datasets([first, second], **({"merge": True} if merge else {}))
    assert len(result) == 2
    assert result[0] is first and result[1] is second


@pytest.mark.parametrize(
    "descriptor",
    [
        False,
        {},
        {"constraints": {"plain": "v"}},
        {"constraints": {"test.key": None}},
        {"constraints": {}, "reason": ""},
        {"constraints": {}, "extra": True},
    ],
)
def test_invalid_plugin_identity_is_not_silently_ignored(identity_provider, descriptor):
    identity_provider(lambda ds: descriptor)
    with pytest.raises(ValueError, match="importer.measurement_identity"):
        merge_datasets([spectrum(), spectrum()], merge=True)


def test_plugin_identity_errors_propagate(identity_provider):
    def provider(dataset):
        raise RuntimeError("invalid scientific metadata")

    identity_provider(provider)
    with pytest.raises(RuntimeError, match="invalid scientific metadata"):
        merge_datasets([spectrum(), spectrum()], merge=True)
    # Explicit opt-out must not consult scientific compatibility providers.
    assert len(merge_datasets([spectrum(), spectrum()], merge=False)) == 2


def test_plugin_identity_in_native_nested_imports(tmp_path, identity_provider):
    identity_provider(
        lambda ds: {"constraints": {"test.nucleus": ds.meta.test_nucleus}}
        if ds.meta.get("test_nucleus")
        else None
    )
    first, unknown, conflict = spectrum(), spectrum(), spectrum()
    first.meta.test_nucleus = "1H"
    conflict.meta.test_nucleus = "13C"
    for name, ds in [("a", unknown), ("b", first), ("c", conflict), ("d", conflict)]:
        ds.dump(tmp_path / f"{name}.scp")
    with pytest.warns(UserWarning, match="test.nucleus differ"):
        result = scp.read_dir(tmp_path, recursive=True, merge=True, sortbydate=False)
    assert len(result) == 2
    assert sum(ds.shape[0] for ds in result) == 4
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


def test_nested_identity_errors_are_not_skipped(tmp_path, identity_provider):
    def provider(dataset):
        if dataset.meta.get("invalid_identity"):
            raise RuntimeError("invalid scientific metadata")

    identity_provider(provider)
    paths = []
    for name in ["invalid", "valid"]:
        directory = tmp_path / name
        directory.mkdir()
        paths.append(directory)
        for index in range(2):
            dataset = spectrum()
            if name == "invalid":
                dataset.meta.invalid_identity = True
            dataset.dump(directory / f"{index}.scp")
    with pytest.raises(RuntimeError, match="invalid scientific metadata"):
        scp.read(paths, merge=True)
