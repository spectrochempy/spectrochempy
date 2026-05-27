# ======================================================================================
# Copyright (C) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the testdata cache marker contract."""

from io import BytesIO
from zipfile import ZipFile

from spectrochempy.application.testdata import _TESTDATA_CACHE_MARKER
from spectrochempy.application.testdata import download_full_testdata_directory


class _Response:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self._content


def _testdata_archive():
    archive = BytesIO()
    with ZipFile(archive, "w") as zipfile:
        zipfile.writestr(
            "spectrochempy_data-master/testdata/example/sample.dat",
            b"complete",
        )
    return archive.getvalue()


def test_legacy_marker_does_not_hide_incomplete_cache(tmp_path, monkeypatch):
    datadir = tmp_path / "testdata"
    datadir.mkdir()
    (datadir / "__downloaded__").touch()
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: _Response(_testdata_archive()),
    )

    download_full_testdata_directory(datadir)

    assert (datadir / "example" / "sample.dat").read_bytes() == b"complete"
    assert (datadir / "__downloaded__").read_text(encoding="utf8") == (
        _TESTDATA_CACHE_MARKER
    )


def test_current_marker_skips_download(tmp_path, monkeypatch):
    datadir = tmp_path / "testdata"
    datadir.mkdir()
    (datadir / "__downloaded__").write_text(_TESTDATA_CACHE_MARKER, encoding="utf8")

    def fail_download(*args, **kwargs):
        raise AssertionError("current testdata cache should not be downloaded again")

    monkeypatch.setattr("requests.get", fail_download)

    download_full_testdata_directory(datadir)
