import numpy as np

import spectrochempy as scp
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.utils.testing import assert_dataset_equal


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
