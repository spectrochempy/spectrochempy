# main module to serve as a basis for mock tests

from spectrochempy import download_iris


def _download_iris():
    return download_iris()


def save_iris_dataset():
    ds = _download_iris()
    ds.save_as("../data/iris_dataset.scp", confirm=False)


if __name__ == "__main__":
    save_iris_dataset()
