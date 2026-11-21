# ruff: noqa: S310, T201
"""Verify and stage Corn inputs outside public caches and artifacts."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED
from zipfile import ZipFile
from zipfile import ZipInfo

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "corn.json"


def prepare(source, directory, env_file):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    content = source.read_bytes()
    if (
        len(content) != manifest["size"]
        or hashlib.sha256(content).hexdigest() != manifest["sha256"]
    ):
        raise ValueError("Corn data does not match the pinned size and SHA-256")
    directory = directory.resolve()
    if any(char in str(directory) for char in "\r\n"):
        raise ValueError("Corn staging path must not contain newlines")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = directory / "corn.mat"
    archive = directory / "corn.zip"
    try:
        raw.write_bytes(content)
        raw.chmod(0o600)
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            member = ZipInfo("corn.mat", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            bundle.writestr(member, content)
        archive.chmod(0o600)
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write(f"SCPY_CORN_DATA={raw}\nSCPY_CORN_ZIP={archive}\n")
    except BaseException:
        raw.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        raise


def prepare_public(directory, env_file):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    request = Request(
        manifest.get("source", "https://www.eigenvector.com/data/Corn/corn.mat"),
        headers={"User-Agent": "SpectroChemPy documentation"},
    )
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310
            content = response.read(manifest["size"] + 1)
    except (URLError, OSError) as error:
        with env_file.open("a", encoding="utf-8") as stream:
            stream.write("SCPY_DOCS_CORN_UNAVAILABLE=1\n")
        print(
            f"::notice::Public Corn data unavailable; skipping Corn-dependent content: {error}"
        )
        return False
    with tempfile.TemporaryDirectory(prefix="scp-public-corn-") as temporary:
        source = Path(temporary) / "corn.mat"
        source.write_bytes(content)
        prepare(source, directory, env_file)
    with env_file.open("a", encoding="utf-8") as stream:
        stream.write("SCPY_DOCS_CORN_UNAVAILABLE=0\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if (args.source is None) != args.public:
        parser.error("provide a source path or --public")
    directory = Path(os.environ["RUNNER_TEMP"]) / "spectrochempy-corn"
    env_file = Path(os.environ["GITHUB_ENV"])
    if args.public:
        prepare_public(directory, env_file)
    else:
        prepare(args.source, directory, env_file)


if __name__ == "__main__":
    main()
