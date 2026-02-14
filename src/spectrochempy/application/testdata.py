# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Testdata downloading."""

import io
from pathlib import Path
from zipfile import ZipFile

import requests


# --------------------------------------------------------------------------------------
# Testdata
# --------------------------------------------------------------------------------------
def download_full_testdata_directory(datadir, force=False):
    """
    Download and extract the full SpectroChemPy test data directory.

    Parameters
    ----------
    datadir : Path
        Target directory where test data should be extracted.
    force : bool, optional
        If True, force re-download even if already downloaded.
        If False, skip download if marker file exists.
    """

    # Marker file used to prevent repeated downloads
    # This avoids re-downloading test data on every docs build.
    downloaded = datadir / "__downloaded__"
    if downloaded.exists() and not force:
        return

    # GitHub archive URL (zip of master branch)
    url = (
        "https://github.com/spectrochempy/"
        "spectrochempy_data/archive/refs/heads/master.zip"
    )

    import tempfile
    import requests
    from zipfile import ZipFile
    from pathlib import Path

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

    # ------------------------------------------------------------------
    # Mark download as completed
    #
    # This prevents re-running the download on subsequent builds
    # unless force=True is explicitly passed.
    # ------------------------------------------------------------------
    downloaded.touch(exist_ok=True)

