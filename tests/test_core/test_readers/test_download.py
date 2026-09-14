# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np

from spectrochempy import NDDataset
from spectrochempy.core.readers import download as download_module


def test_download_nist_replaces_import_history_without_losing_later_entries(
    monkeypatch, tmp_path
):
    class Response:
        content = b"##TITLE=fake"

        @staticmethod
        def iter_content():
            yield b"##TITLE=fake"

    expected_data = np.array([1.0, 2.0])
    source_holder = {}

    def fake_read_jcamp(filename):
        dataset = NDDataset(expected_data, title="fixture", origin="synthetic")
        dataset.meta.operator = "tester"
        dataset.history = "Imported from jdx file"
        dataset.history = "Vendor processing retained"
        source_holder["dataset"] = dataset
        source_holder["first_date"] = dataset._history[0][0]
        return dataset

    monkeypatch.setattr(
        download_module.requests, "get", lambda *args, **kwargs: Response()
    )
    monkeypatch.setattr(download_module, "read_jcamp", fake_read_jcamp)
    monkeypatch.chdir(tmp_path)

    result = download_module.download_nist_ir("7732-18-5", index=0)

    assert result is source_holder["dataset"]
    assert result._history[0][0] == source_holder["first_date"]
    assert len(result.history) == 2
    assert "Downloaded from NIST:" in result.history[0]
    assert "Imported from jdx file" not in result.history[0]
    assert "Vendor processing retained" in result.history[1]
    assert all(isinstance(entry, str) for entry in result.history)
    np.testing.assert_array_equal(result.data, expected_data)
    assert result.title == "fixture"
    assert result.origin == "synthetic"
    assert result.meta.operator == "tester"


# def test_download_nist():
#     CAS = "7732-18-5"  # WATER

#     try:
#         ds = scp.download_nist_ir(CAS)
#     except OSError:
#         # no internet connection or server down
#         # (should not fail test)
#         return

#     assert len(ds) == 2

# skip next tests  as the NIST server blocks  the request
# when several requests are made in a short time

# ds = scp.download_nist_ir(CAS, index=0)
# assert ds.name == "Water"
#
# ds = scp.download_nist_ir(CAS, index=[0, 1])
# assert len(ds) == 2
#
# ds = scp.download_nist_ir(CAS, index=2)
# assert ds is None
#
# ds = scp.download_nist_ir(CAS, index=[0, 1, 2])
# assert len(ds) == 2
#
# CAS = 2146363  # Acenaphthylene, dodecahydro-
# ds = scp.download_nist_ir(CAS)
# assert ds is None
