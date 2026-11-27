"""Qualify the synthetic Corn schema, workflows and documentation boundary."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import yaml

import spectrochempy as scp
from docs import corn_data

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "synthetic_corn", ROOT / ".github/workflows/scripts/prepare_synthetic_corn.py"
)
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Synthetic Corn workflows must not open network connections")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    return synthetic.prepare(tmp_path / "data", tmp_path / "env")


def test_reproducible_mat_and_zip(inputs, tmp_path):
    raw, archive = inputs
    second = synthetic.prepare(tmp_path / "second", tmp_path / "env2")
    assert raw.read_bytes() == second[0].read_bytes()
    assert archive.read_bytes() == second[1].read_bytes()
    with ZipFile(archive) as bundle:
        assert bundle.namelist() == [raw.name]
        assert bundle.read(raw.name) == raw.read_bytes()
    assert "SCPY_CORN_SYNTHETIC=1" in (tmp_path / "env").read_text()


@pytest.mark.parametrize("reader", [scp.read, scp.read_matlab])
def test_dso_schema(inputs, reader):
    datasets = reader(inputs[0], merge=False)
    assert [(d.name, d.shape) for d in datasets] == [
        ("m5nbs", (3, 700)),
        ("mp5nbs", (4, 700)),
        ("mp6nbs", (4, 700)),
        ("propvals", (80, 4)),
        ("m5spec", (80, 700)),
        ("mp5spec", (80, 700)),
        ("mp6spec", (80, 700)),
    ]
    assert datasets[3][:, "Moisture"].shape == (80, 1)
    np.testing.assert_array_equal(datasets[4].x.data, np.arange(1100, 2500, 2))
    assert all(synthetic.MARKER in d.description for d in datasets)
    zipped = scp.read(inputs[1], merge=False)
    assert [d.name for d in zipped] == [d.name for d in datasets]
    for actual, expected in zip(zipped, datasets, strict=True):
        np.testing.assert_array_equal(actual.data, expected.data)


def test_learnable_held_out_and_fold_local_workflows(inputs):
    datasets = scp.read(inputs[0], merge=False)
    X, y = datasets[4], datasets[3][:, "Moisture"].squeeze()
    model = scp.PLSRegression(n_components=5)
    model.fit(X[:57], y[:57])
    predicted = model.predict(X[57:]).data.squeeze()
    observed = y[57:].data.squeeze()
    assert np.sqrt(np.mean((predicted - observed) ** 2)) < 0.15
    for estimator in (
        scp.PLSRegression(n_components=5),
        scp.Pipeline(
            [
                ("msc", scp.MSCTransformer(dim="y")),
                ("pls", scp.PLSRegression(n_components=5)),
            ]
        ),
    ):
        result = scp.cross_validate(
            estimator,
            X,
            y,
            cv=scp.KFold(n_splits=5, shuffle=True, random_state=7),
            metrics=("rmsecv", "r2"),
        )
        assert float(result.metric("r2").values.data.squeeze()) > 0.5
        assert np.isfinite(result.oof_predictions.data).all()


@pytest.mark.parametrize("kind", ["environment", "raw", "zip"])
def test_documentation_rejects_synthetic_inputs(inputs, monkeypatch, kind):
    for key in ("SCPY_CORN_SYNTHETIC", "SCPY_CORN_DATA", "SCPY_CORN_ZIP"):
        monkeypatch.delenv(key, raising=False)
    key, value = {
        "environment": ("SCPY_CORN_SYNTHETIC", "1"),
        "raw": ("SCPY_CORN_DATA", str(inputs[0])),
        "zip": ("SCPY_CORN_ZIP", str(inputs[1])),
    }[kind]
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="Synthetic Corn"):
        corn_data.require_real_data()


def test_generator_is_pr_test_only():
    workflow = yaml.safe_load((ROOT / ".github/workflows/test_package.yml").read_text())
    steps = [s for job in workflow["jobs"].values() for s in job.get("steps", [])]
    generator = next(
        s for s in steps if "prepare_synthetic_corn.py" in s.get("run", "")
    )
    assert generator["if"] == "github.event_name == 'pull_request'"
    cleanup = next(
        s for s in steps if s.get("name") == "Remove private Corn working copies"
    )
    assert cleanup["if"] == "always()"
    assert "spectrochempy-synthetic-corn" in cleanup["run"]
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        if path.name != "test_package.yml":
            assert "prepare_synthetic_corn.py" not in path.read_text()


def test_documentation_allows_unmarked_real_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("SCPY_CORN_SYNTHETIC", raising=False)
    monkeypatch.delenv("SCPY_CORN_ZIP", raising=False)
    raw = tmp_path / "corn.mat"
    raw.write_bytes(b"MATLAB 5.0 MAT-file, unmarked input")
    monkeypatch.setenv("SCPY_CORN_DATA", str(raw))
    corn_data.require_real_data()


def test_rejects_environment_line_injection(tmp_path):
    with pytest.raises(ValueError, match="newlines"):
        synthetic.prepare(tmp_path / "bad\nINJECT=1", tmp_path / "env")
    assert not (tmp_path / "env").exists()
