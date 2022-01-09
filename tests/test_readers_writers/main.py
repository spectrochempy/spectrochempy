# main module to serve as a basis for mock tests

import pathlib

# iris download
# -------------

from spectrochempy import download_iris


def _download_iris():
    return download_iris()


def get_path():
    for p in pathlib.Path().cwd().parents:
        if p.stem == "spectrochempy":
            break
    return p


def save_iris_dataset():
    ds = _download_iris()

    path = get_path()
    path = path / "tests/data/"
    path.mkdir(parents=True, exist_ok=True)
    ds.save_as(path / "iris_dataset.scp", confirm=False)


if __name__ == "__main__":
    save_iris_dataset()
