"""Generate synthetic MATLAB DSO inputs for PR workflow smoke tests only."""

import argparse
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED
from zipfile import ZipFile
from zipfile import ZipInfo

import numpy as np
from scipy.io import savemat

MARKER = "SPECTROCHEMPY SYNTHETIC CORN FIXTURE - NOT EXPERIMENTAL DATA"


def _dso(name, values, wavelengths=None, labels=None):
    axis = np.empty((2, 2), dtype=object)
    axis[:] = ""
    if wavelengths is not None:
        axis[1, 0] = wavelengths.reshape(1, -1)
        axis[1, 1] = "Wavelength"
    label = np.empty((2, 2), dtype=object)
    label[:] = ""
    if labels is not None:
        label[1, 0] = np.asarray(labels)
    return {
        "name": name,
        "type": "data",
        "author": "SpectroChemPy synthetic test generator",
        "date": [2000, 1, 1, 0, 0, 0],
        "moddate": [2000, 1, 1, 0, 0, 0],
        "imagesize": [],
        "data": values,
        "label": label,
        "axisscale": axis,
        "description": MARKER,
        "history": np.array([[MARKER]], dtype=object),
    }


def prepare(directory, env_file):
    """Write reproducible synthetic inputs without reading or downloading Corn."""
    directory = directory.resolve()
    if any(char in str(directory) for char in "\r\n"):
        raise ValueError("Corn staging path must not contain newlines")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1992)
    wavelength = np.arange(1100.0, 2500.0, 2.0)
    latent = rng.uniform(0.2, 1.0, (80, 6))
    centers = np.array([1210, 1410, 1650, 1910, 2140, 2330])
    bands = np.exp(-0.5 * ((wavelength - centers[:, None]) / 45.0) ** 2)
    spectra = 0.3 + latent @ bands + rng.normal(0, 0.002, (80, 700))
    properties = np.column_stack(
        [
            8 + 4 * latent[:, 0] + latent[:, 1],
            2 + latent[:, 2],
            7 + 3 * latent[:, 3],
            60 + 8 * latent[:, 4],
        ]
    )
    properties += rng.normal(0, 0.01, properties.shape)
    data = {
        "m5spec": _dso("m5spec", spectra, wavelength),
        "mp5spec": _dso("mp5spec", spectra * 1.02 + 0.01, wavelength),
        "mp6spec": _dso("mp6spec", spectra * 0.98 + 0.02, wavelength),
        "propvals": _dso(
            "propvals", properties, labels=["Moisture", "Oil", "Protein", "Starch"]
        ),
        "m5nbs": _dso("m5nbs", bands[:3], wavelength),
        "mp5nbs": _dso("mp5nbs", bands[:4] * 1.02, wavelength),
        "mp6nbs": _dso("mp6nbs", bands[:4] * 0.98, wavelength),
    }
    raw = directory / "synthetic-corn.mat"
    archive = directory / "synthetic-corn.zip"
    savemat(raw, data, do_compression=True)
    content = raw.read_bytes()
    content = MARKER.encode("ascii").ljust(116, b" ") + content[116:]
    raw.write_bytes(content)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        member = ZipInfo("synthetic-corn.mat", date_time=(1980, 1, 1, 0, 0, 0))
        member.compress_type = ZIP_DEFLATED
        bundle.writestr(member, content)
    with env_file.open("a", encoding="utf-8") as stream:
        stream.write(
            f"SCPY_CORN_DATA={raw}\nSCPY_CORN_ZIP={archive}\nSCPY_CORN_SYNTHETIC=1\n"
        )
    return raw, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(os.environ.get("RUNNER_TEMP", "."))
        / "spectrochempy-synthetic-corn",
    )
    parser.add_argument("--env-file", type=Path, default=os.environ.get("GITHUB_ENV"))
    args = parser.parse_args()
    if args.env_file is None:
        parser.error("provide --env-file outside GitHub Actions")
    prepare(args.directory, args.env_file)


if __name__ == "__main__":
    main()
