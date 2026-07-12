# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Testdata downloading."""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Testdata
# --------------------------------------------------------------------------------------
_TESTDATA_CACHE_MARKER = "spectrochempy-testdata-v1\n"
_TESTDATA_EXTRA_CACHE_MARKER = "spectrochempy-testdata-extra-v1\n"
_EXTRA_REPO_URL = "https://github.com/spectrochempy/spectrochempy_data.git"
_EXTRA_BRANCH = "data-extra"


def download_full_testdata_directory(datadir, force=False):
    """
    Download and extract the full SpectroChemPy test data directory.

    Parameters
    ----------
    datadir : Path
        Target directory where test data should be extracted.
    force : bool, optional
        If True, force re-download even if already downloaded.
        If False, skip download if a current cache marker exists.
    """

    # Only a marker created by this extraction contract is trusted. Legacy empty
    # markers may represent partial caches restored by CI.
    downloaded = datadir / "__downloaded__"
    if (
        downloaded.exists()
        and downloaded.read_text(encoding="utf8") == _TESTDATA_CACHE_MARKER
        and not force
    ):
        return

    # GitHub archive URL (zip of master branch)
    url = (
        "https://github.com/spectrochempy/"
        "spectrochempy_data/archive/refs/heads/master.zip"
    )

    import tempfile
    from zipfile import ZipFile

    import requests

    # ------------------------------------------------------------------
    # Perform HTTP request
    #
    # We use:
    # - stream=True to avoid loading the entire archive in memory
    # - timeout=(10, 120):
    #     * 10 seconds to establish connection
    #     * 120 seconds allowed per read block
    #
    # The previous timeout=10 caused intermittent ReadTimeout errors
    # when GitHub throttled the download.
    # ------------------------------------------------------------------
    resp = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(10, 120),
    )

    # Raise an explicit error if HTTP status is not 200
    resp.raise_for_status()

    # ------------------------------------------------------------------
    # Stream archive to a temporary file
    #
    # We avoid using resp.content because:
    # - It forces the entire download into memory
    # - It increases risk of timeout
    # - It may produce truncated archives under slow connections
    #
    # Streaming chunk-by-chunk is safer and more memory-efficient.
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                tmp.write(chunk)
        tmp_path = tmp.name

    # ------------------------------------------------------------------
    # Extract only the testdata/ subdirectory
    #
    # The GitHub archive has structure:
    #   spectrochempy_data-master/...
    #
    # We skip:
    #   - directories
    #   - files not under testdata/
    #
    # Then recreate the internal structure inside `datadir`.
    # ------------------------------------------------------------------
    try:
        with ZipFile(tmp_path) as zipfile:
            for name in zipfile.namelist():
                # Skip directories and non-testdata files
                if name.endswith("/") or "testdata/" not in name:
                    continue

                # Read file contents from archive
                uncompressed = zipfile.read(name)

                # Remove first two path components:
                # spectrochempy_data-master/<something>/...
                p = list(Path(name).parts)[2:]
                dst = datadir.joinpath("/".join(p))

                # Ensure target directory exists
                dst.parent.mkdir(parents=True, exist_ok=True)

                # Write extracted file
                dst.write_bytes(uncompressed)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Mark download as completed
    #
    # This prevents re-running the download on subsequent builds
    # unless force=True is explicitly passed.
    # ------------------------------------------------------------------
    downloaded.write_text(_TESTDATA_CACHE_MARKER, encoding="utf8")


def download_extra_testdata(datadir=None, force=False):
    """Download extra test data from the ``data-extra`` branch.

    Extra datasets (agilent, jeol, bruker_3d, simpson, tecmag) are stored on a
    separate branch to keep the main download lightweight.

    Parameters
    ----------
    datadir : Path or str, optional
        Target directory.  Defaults to ``~/.spectrochempy/testdata-extra/``.
    force : bool, optional
        If True, force re-download even if already downloaded.
    """
    import shutil
    import subprocess

    if datadir is None:
        datadir = Path.home() / ".spectrochempy" / "testdata-extra"
    else:
        datadir = Path(datadir)

    marker = datadir / "__downloaded_extra__"
    if (
        marker.exists()
        and marker.read_text(encoding="utf8") == _TESTDATA_EXTRA_CACHE_MARKER
        and not force
    ):
        return datadir

    # Remove partial clone if marker is missing
    if not marker.exists() and datadir.exists() and any(datadir.iterdir()):
        shutil.rmtree(datadir)

    datadir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--branch",
        _EXTRA_BRANCH,
        "--depth",
        "1",
        "--single-branch",
        _EXTRA_REPO_URL,
        str(datadir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = f"git clone failed (exit {result.returncode}):\n{result.stderr}"
        raise RuntimeError(msg)

    # Remove .git directory — we only need the data files
    git_dir = datadir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    # Mark download as completed
    marker.write_text(_TESTDATA_EXTRA_CACHE_MARKER, encoding="utf8")

    return datadir
