# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
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
    # this process is relatively long, so we do not want to do it several time:
    downloaded = datadir / "__downloaded__"
    if downloaded.exists() and not force:
        return

    url = "https://github.com/spectrochempy/spectrochempy_data/archive/refs/heads/master.zip"

    resp = requests.get(url, stream=True, allow_redirects=True, timeout=10)
    zipfile = ZipFile(io.BytesIO(resp.content))
    files = [zipfile.open(file_name) for file_name in zipfile.namelist()]

    for file in files:
        name = file.name
        if name.endswith("/") or "testdata/" not in name:  # dir
            continue
        uncompressed = zipfile.read(name)
        p = list(Path(name).parts)[2:]
        dst = datadir.joinpath("/".join(p))
        if not dst.parent.exists():
            # create the eventually missing subdirectory
            dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(uncompressed)

    # write the "__downloaded__" file to avoid this process to run several file.
    downloaded.touch(exist_ok=True)
