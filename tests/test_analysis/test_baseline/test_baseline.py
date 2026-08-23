import warnings

import numpy as np
import pytest
from scipy.sparse import SparseEfficiencyWarning

import spectrochempy as scp
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.utils.testing import assert_dataset_equal


def _make_simple_baseline_dataset(
    n_points=11, *, descending=False, units="cm^-1", dataset_units="absorbance"
):
    x = np.linspace(1000.0, 2000.0, n_points)
    if descending:
        x = x[::-1]

    # Exact polynomial support keeps the validation tests focused on API
    # contracts rather than numeric quality.
    data = 0.002 * x + 0.5

    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(x, title="wavenumber", units=units)],
        units=dataset_units,
        title="simple baseline dataset",
    )
    dataset.name = f"simple_baseline_{n_points}"
    return dataset


def _make_shape_probe_dataset(
    n_points=64,
    *,
    n_rows=None,
    descending=False,
    units="cm^-1",
    dataset_units="absorbance",
):
    x = np.linspace(1000.0, 2000.0, n_points)
    if descending:
        x = x[::-1]

    base = 0.002 * x + 0.5 + 0.1 * np.sin(np.linspace(0.0, 4.0, n_points))
    if n_rows is None:
        data = base
        coordset = [scp.Coord(x, title="wavenumber", units=units)]
        name = f"shape_probe_1d_{n_points}"
    else:
        data = np.vstack([base + 0.01 * i for i in range(n_rows)])
        coordset = [
            scp.Coord(np.arange(n_rows, dtype=float), title="row", units=None),
            scp.Coord(x, title="wavenumber", units=units),
        ]
        name = f"shape_probe_{n_rows}x{n_points}"

    dataset = scp.NDDataset(
        data,
        coordset=coordset,
        units=dataset_units,
        title="baseline shape probe dataset",
    )
    dataset.name = name
    return dataset


def test_baseline_fit_1d(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    blc = Baseline()
    blc.fit(dataset)
    baseline = blc.baseline
    corrected = blc.transform()

    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))

    assert baseline.dims == dataset.dims
    assert baseline.units == dataset.units


def test_baseline_fit_2d(synthetic_2d_baseline_dataset):
    dataset, _, _ = synthetic_2d_baseline_dataset

    blc = Baseline()
    blc.fit(dataset)
    baseline = blc.baseline
    corrected = blc.corrected

    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        ("polynomial", {"order": 3}),
        ("asls", {"lamb": 1e5, "asymmetry": 0.05}),
        ("snip", {"snip_width": 15}),
        ("rubberband", {}),
    ],
)
@pytest.mark.parametrize("descending", [False, True])
def test_baseline_corrected_preserves_shape_for_strict_1d_inputs(
    model, kwargs, descending
):
    dataset = _make_shape_probe_dataset(descending=descending)
    original = dataset.copy()

    blc = Baseline(model=model, **kwargs)
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.corrected

    assert baseline.shape == dataset.shape
    assert corrected.shape == dataset.shape
    assert baseline.dims == dataset.dims
    assert corrected.dims == dataset.dims
    assert baseline.units == dataset.units
    assert corrected.units == dataset.units
    np.testing.assert_allclose(baseline.x.data, dataset.x.data)
    np.testing.assert_allclose(corrected.x.data, dataset.x.data)
    assert baseline.x.is_descendant == dataset.x.is_descendant
    assert corrected.x.is_descendant == dataset.x.is_descendant
    np.testing.assert_allclose(
        np.asarray(corrected.data),
        np.asarray(dataset.data) - np.asarray(baseline.data),
    )
    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        ("polynomial", {"order": 3}),
        ("asls", {"lamb": 1e5, "asymmetry": 0.05}),
        ("snip", {"snip_width": 15}),
        ("rubberband", {}),
    ],
)
@pytest.mark.parametrize("n_rows", [1, 3])
def test_baseline_corrected_preserves_shape_for_2d_inputs(model, kwargs, n_rows):
    dataset = _make_shape_probe_dataset(n_rows=n_rows)
    original = dataset.copy()

    blc = Baseline(model=model, **kwargs)
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.corrected

    assert baseline.shape == dataset.shape
    assert corrected.shape == dataset.shape
    assert baseline.dims == dataset.dims
    assert corrected.dims == dataset.dims
    assert baseline.units == dataset.units
    assert corrected.units == dataset.units
    np.testing.assert_allclose(baseline.x.data, dataset.x.data)
    np.testing.assert_allclose(corrected.x.data, dataset.x.data)
    np.testing.assert_allclose(baseline.y.data, dataset.y.data)
    np.testing.assert_allclose(corrected.y.data, dataset.y.data)
    np.testing.assert_allclose(
        np.asarray(corrected.data),
        np.asarray(dataset.data) - np.asarray(baseline.data),
    )
    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (scp.basc, {}),
        (scp.asls, {"lamb": 1e5, "asymmetry": 0.05}),
        (lambda dataset: dataset.get_baseline(model="polynomial", order=3), {}),
    ],
)
@pytest.mark.parametrize(
    ("n_rows", "descending"),
    [
        (None, False),
        (None, True),
        (1, False),
    ],
)
def test_baseline_public_helpers_preserve_public_shape(
    func, kwargs, n_rows, descending
):
    dataset = _make_shape_probe_dataset(n_rows=n_rows, descending=descending)
    original = dataset.copy()

    output = func(dataset, **kwargs) if kwargs else func(dataset)

    assert output.shape == dataset.shape
    assert output.dims == dataset.dims
    assert output.units == dataset.units
    np.testing.assert_allclose(output.x.data, dataset.x.data)
    assert output.x.is_descendant == dataset.x.is_descendant
    if n_rows is not None:
        np.testing.assert_allclose(output.y.data, dataset.y.data)
    assert_dataset_equal(dataset, original)


def test_baseline_corrected_preserves_mask_and_shape_for_strict_1d_snip():
    dataset = _make_shape_probe_dataset()
    dataset[1400.0:1500.0] = scp.MASKED
    original = dataset.copy()
    expected_mask = np.asarray(dataset.mask).copy()

    blc = Baseline(model="snip", snip_width=15)
    blc.fit(dataset)
    corrected = blc.corrected

    assert corrected.shape == dataset.shape
    assert corrected.dims == dataset.dims
    assert np.asarray(corrected.mask).shape == expected_mask.shape
    assert np.array_equal(np.asarray(corrected.mask), expected_mask)
    assert np.array_equal(np.asarray(dataset.mask), np.asarray(original.mask))
    assert_dataset_equal(dataset, original)


def test_baseline_polynomial_recovers_known_1d_baseline(
    synthetic_1d_baseline_dataset,
):
    dataset, true_baseline, _ = synthetic_1d_baseline_dataset

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = 3
    blc.fit(dataset)

    estimated = blc.baseline.data
    diff = np.abs(estimated - true_baseline)
    assert blc.baseline.shape == dataset.shape
    assert np.mean(diff) < 0.25
    assert np.max(diff) < 0.35


def test_baseline_asls_1d(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    blc = Baseline(log_level="INFO")
    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.transform()
    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))


def test_baseline_asls_1d_emits_no_sparse_efficiency_warning(
    synthetic_1d_baseline_dataset,
):
    dataset, _, _ = synthetic_1d_baseline_dataset

    blc = Baseline(log_level="WARNING")
    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        blc.fit(dataset)

    sparse_warnings = [
        warning
        for warning in recorded
        if issubclass(warning.category, SparseEfficiencyWarning)
    ]
    assert sparse_warnings == []


def test_baseline_asls_2d(synthetic_2d_baseline_dataset):
    dataset, _, _ = synthetic_2d_baseline_dataset

    blc = Baseline(log_level="INFO")
    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.transform()
    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))


def test_baseline_masked_data(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    dataset[3000.0:2000.0] = scp.MASKED

    blc = Baseline()
    blc.fit(dataset)
    baseline = blc.baseline
    corrected = blc.transform()

    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))

    blc.model = "asls"
    blc.mu = 0.5 * 10**9
    blc.asymmetry = 0.001
    blc.fit(dataset)
    baseline = blc.baseline
    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))


def test_baseline_pchip_smoke(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = "pchip"
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert np.all(np.isfinite(blc.baseline.data))

    blc.order = 3
    blc.fit(dataset)
    assert blc.baseline.shape == dataset.shape
    assert np.all(np.isfinite(blc.baseline.data))


def test_baseline_multivariate_svd_smoke(synthetic_2d_baseline_dataset):
    dataset, _, _ = synthetic_2d_baseline_dataset

    blc = Baseline()
    blc.multivariate = True
    blc.model = "polynomial"
    blc.order = "pchip"
    blc.n_components = 3
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert np.all(np.isfinite(blc.baseline.data))
    assert np.all(np.isfinite(blc.transform().data))


def test_baseline_multivariate_nmf_smoke(synthetic_2d_baseline_dataset):
    dataset, _, _ = synthetic_2d_baseline_dataset

    blc = Baseline()
    blc.multivariate = "nmf"
    blc.model = "polynomial"
    blc.order = 6
    blc.n_components = 3
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert np.all(np.isfinite(blc.baseline.data))
    assert np.all(np.isfinite(blc.transform().data))


def test_baseline_sequential_asls(synthetic_2d_baseline_dataset):
    dataset, _, _ = synthetic_2d_baseline_dataset

    dataset[:, 3000.0:2000.0] = scp.MASKED

    blc = Baseline(log_level="INFO")
    blc.multivariate = False
    blc.model = "asls"
    blc.mu = 10**8
    blc.asymmetry = 0.002
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.corrected
    assert baseline.shape == dataset.shape
    assert corrected.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))


def test_baseline_polynomial_with_ranges(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = 2
    blc.ranges = [[3800.0, 3600.0], [1800.0, 1200.0]]
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert np.all(np.isfinite(blc.baseline.data))


@pytest.mark.parametrize(
    ("n_points", "order", "should_fit"),
    [
        (1, "constant", True),
        (1, 0, True),
        (1, 1, False),
        (2, 1, True),
        (2, 3, False),
        (3, 1, True),
        (3, 3, False),
    ],
)
def test_baseline_polynomial_short_dataset_validation(n_points, order, should_fit):
    dataset = _make_simple_baseline_dataset(n_points)

    blc = Baseline(model="polynomial", order=order)

    if should_fit:
        blc.fit(dataset)
        assert np.all(np.isfinite(np.asarray(blc.baseline.data)))
        assert blc.baseline.units == dataset.units
    else:
        with pytest.raises(ValueError, match="too short"):
            blc.fit(dataset)


def test_baseline_polynomial_without_ranges_and_without_limits_errors():
    dataset = _make_simple_baseline_dataset()
    original = dataset.copy()

    blc = Baseline(model="polynomial", order=1, include_limits=False)

    with pytest.raises(ValueError, match="No baseline support ranges were selected"):
        blc.fit(dataset)

    assert_dataset_equal(dataset, original)


def test_baseline_polynomial_constant_order_accepts_single_point_range():
    dataset = _make_simple_baseline_dataset()

    blc = Baseline(model="polynomial", order="constant", include_limits=False)
    blc.ranges = [[1500.0, 1500.0]]
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert blc.baseline.units == dataset.units
    assert blc.corrected.units == dataset.units


@pytest.mark.parametrize(
    ("descending", "ranges", "should_fit"),
    [
        (False, [[1500.0, 1500.0]], False),
        (True, [[1500.0, 1500.0]], False),
        (False, [[1400.0, 1400.0], [1600.0, 1600.0]], True),
        (True, [[1400.0, 1400.0], [1600.0, 1600.0]], True),
    ],
)
def test_baseline_polynomial_pchip_support_validation(descending, ranges, should_fit):
    dataset = _make_simple_baseline_dataset(descending=descending)
    original = dataset.copy()

    blc = Baseline(model="polynomial", order="pchip", include_limits=False)
    blc.ranges = ranges

    if should_fit:
        blc.fit(dataset)
        assert blc.baseline.shape == dataset.shape
        assert blc.baseline.units == dataset.units
        assert blc.corrected.units == dataset.units
        assert blc.baseline.x.is_descendant == dataset.x.is_descendant
    else:
        with pytest.raises(ValueError, match="support is too small"):
            blc.fit(dataset)

    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([[1500.0, 1500.0]], "support is too small"),
        ([[1500.0, 1500.1]], "support is too small"),
        ([[1500.0, 1500.0], [1700.0, 1700.0]], "support is too small"),
    ],
)
def test_baseline_polynomial_rejects_insufficient_support_ranges(ranges, message):
    dataset = _make_simple_baseline_dataset()
    original = dataset.copy()

    blc = Baseline(model="polynomial", order=3, include_limits=False)
    blc.ranges = ranges

    with pytest.raises(ValueError, match=message):
        blc.fit(dataset)

    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize("descending", [False, True])
def test_baseline_polynomial_rejects_ranges_outside_domain(descending):
    dataset = _make_simple_baseline_dataset(descending=descending)
    original = dataset.copy()

    blc = Baseline(model="polynomial", order=1, include_limits=False)
    blc.ranges = [[2500.0, 2600.0]]

    with pytest.raises(ValueError, match="do not intersect the coordinate domain"):
        blc.fit(dataset)

    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize("descending", [False, True])
def test_baseline_polynomial_rejects_mixed_in_and_out_of_domain_ranges(descending):
    dataset = _make_simple_baseline_dataset(descending=descending)
    original = dataset.copy()

    blc = Baseline(model="polynomial", order=1, include_limits=False)
    blc.ranges = [[1100.0, 1200.0], [2500.0, 2600.0]]

    with pytest.raises(
        ValueError, match="Some requested baseline ranges do not intersect"
    ):
        blc.fit(dataset)

    assert_dataset_equal(dataset, original)


@pytest.mark.parametrize("descending", [False, True])
def test_baseline_polynomial_accepts_partially_out_of_domain_ranges(descending):
    dataset = _make_simple_baseline_dataset(descending=descending)
    original = dataset.copy()

    blc = Baseline(model="polynomial", order=1, include_limits=False)
    blc.ranges = [[900.0, 1200.0], [1800.0, 2100.0]]
    blc.fit(dataset)

    assert blc.baseline.shape == dataset.shape
    assert blc.baseline.units == dataset.units
    assert blc.corrected.units == dataset.units
    assert blc.baseline.x.is_descendant == dataset.x.is_descendant
    assert_dataset_equal(dataset, original)


def test_preprocessing_nddataset_methods(synthetic_1d_baseline_dataset):
    dataset, _, _ = synthetic_1d_baseline_dataset

    dataset[3000.0:2000.0] = scp.MASKED

    baseline = dataset.get_baseline()
    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))

    baseline_asls = dataset.get_baseline(model="asls", lamb=10**8, asymmetry=0.002)
    assert baseline_asls.shape == dataset.shape
    assert np.all(np.isfinite(baseline_asls.data))

    ndpcor_asls = scp.asls(dataset, lamb=10**8, asymmetry=0.002)
    assert_dataset_equal(ndpcor_asls, dataset - baseline_asls)

    ndpcor_snip = scp.snip(dataset, snip_width=150)
    baseline_snip = dataset.get_baseline(model="snip", snip_width=150)
    assert_dataset_equal(ndpcor_snip.squeeze(), dataset - baseline_snip)


def test_baseline_ms_profile(synthetic_ms_like_dataset):
    dataset, _, _ = synthetic_ms_like_dataset

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = 2
    blc.fit(dataset)

    baseline = blc.baseline
    corrected = blc.corrected
    assert baseline.shape == dataset.shape
    assert np.all(np.isfinite(baseline.data))
    assert np.all(np.isfinite(corrected.data))


def test_baseline_preserves_mask_2d(synthetic_2d_baseline_dataset):
    # #1097: masking a spectral region must survive baseline correction unchanged.
    # The existing masked-baseline tests only check shape and finiteness; none assert
    # that the mask locations themselves are preserved on the baseline/corrected output.
    dataset, _, _ = synthetic_2d_baseline_dataset

    dataset[:, 3000.0:2000.0] = scp.MASKED
    expected_mask = np.asarray(dataset.mask).copy()
    assert expected_mask.any()

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = 3
    blc.fit(dataset)
    baseline = blc.baseline
    corrected = blc.transform()

    # mask locations remain unchanged on both the baseline and the corrected dataset
    assert np.array_equal(np.asarray(baseline.mask), expected_mask)
    assert np.array_equal(np.asarray(corrected.mask), expected_mask)

    # units, dimensions and shape are preserved
    assert corrected.shape == dataset.shape
    assert corrected.units == dataset.units
    assert baseline.units == dataset.units
    assert corrected.dims == dataset.dims

    # no values are introduced into the masked region: unmasked data stays finite
    unmasked = ~np.asarray(corrected.mask)
    assert np.all(np.isfinite(corrected.data[unmasked]))
    assert np.all(np.isfinite(baseline.data[unmasked]))


def test_baseline_preserves_mask_1d(synthetic_1d_baseline_dataset):
    # #1097, 1D case: a masked region in a 1D spectrum survives baseline correction.
    # The dataset is processed as a single-row 2D internally, but the mask is restored
    # at the same coordinate positions.
    dataset, _, _ = synthetic_1d_baseline_dataset

    dataset[3000.0:2000.0] = scp.MASKED
    expected_positions = np.flatnonzero(np.asarray(dataset.mask).ravel())
    assert expected_positions.size

    blc = Baseline()
    blc.model = "polynomial"
    blc.order = 3
    blc.fit(dataset)
    corrected = blc.transform()

    out_positions = np.flatnonzero(np.asarray(corrected.mask).ravel())
    assert np.array_equal(out_positions, expected_positions)
    assert corrected.units == dataset.units

    unmasked = ~np.asarray(corrected.mask).ravel()
    assert np.all(np.isfinite(corrected.data.ravel()[unmasked]))


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        ("polynomial", {"order": 3}),
        ("asls", {"lamb": 10**8, "asymmetry": 0.002}),
        ("snip", {"snip_width": 40}),
        ("rubberband", {}),
    ],
)
def test_baseline_models_preserve_mask_2d(synthetic_2d_baseline_dataset, model, kwargs):
    """Core baseline models preserve masked regions on baseline and corrected output."""
    dataset, _, _ = synthetic_2d_baseline_dataset
    dataset[:, 3000.0:2000.0] = scp.MASKED
    expected_mask = np.asarray(dataset.mask).copy()

    blc = Baseline()
    blc.model = model
    for key, value in kwargs.items():
        setattr(blc, key, value)

    blc.fit(dataset)
    baseline = blc.baseline
    corrected = blc.transform()

    assert np.array_equal(np.asarray(baseline.mask), expected_mask)
    assert np.array_equal(np.asarray(corrected.mask), expected_mask)

    unmasked = ~np.asarray(corrected.mask)
    assert np.all(np.isfinite(corrected.data[unmasked]))
    assert np.all(np.isfinite(baseline.data[unmasked]))


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (scp.asls, {"lamb": 10**8, "asymmetry": 0.002}),
        (scp.snip, {"snip_width": 40}),
        (scp.rubberband, {}),
    ],
)
def test_baseline_wrapper_functions_preserve_mask_1d(
    synthetic_1d_baseline_dataset, func, kwargs
):
    """Public baseline-correction helpers preserve masks on corrected output."""
    dataset, _, _ = synthetic_1d_baseline_dataset
    dataset[3000.0:2000.0] = scp.MASKED
    expected_positions = np.flatnonzero(np.asarray(dataset.mask).ravel())

    corrected = func(dataset, **kwargs)

    out_positions = np.flatnonzero(np.asarray(corrected.mask).ravel())
    assert np.array_equal(out_positions, expected_positions)

    unmasked = ~np.asarray(corrected.mask).ravel()
    assert np.all(np.isfinite(corrected.data.ravel()[unmasked]))
