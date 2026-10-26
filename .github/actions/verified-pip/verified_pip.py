"""Securely bootstrap pip when the runner has an affected version."""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

MINIMUM_VERSION = (26, 2, 2)
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/"
    "pip-26.2.2-py3-none-any.whl"
)
WHEEL_SHA256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"


def version_tuple(version):
    return tuple(int(part) for part in version.split(".")[:3])


def pip_version():
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "--version"],  # noqa: S603
        check=True,
        capture_output=True,
        text=True,
    )
    return version_tuple(result.stdout.split()[1])


def main():
    if pip_version() < MINIMUM_VERSION:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / WHEEL_URL.rsplit("/", maxsplit=1)[-1]
            urlretrieve(WHEEL_URL, wheel)  # noqa: S310
            digest = hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest()
            if digest != WHEEL_SHA256:
                raise RuntimeError(
                    "The verified pip bootstrap wheel has an invalid SHA-256."
                )
            subprocess.run(  # noqa: S603
                [  # noqa: S603
                    sys.executable,  # noqa: S603
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--upgrade",
                    str(wheel),
                ],
                check=True,
            )
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],  # noqa: S603
        check=True,
    )


if __name__ == "__main__":
    main()
