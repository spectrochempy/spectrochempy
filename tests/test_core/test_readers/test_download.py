# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa


# import spectrochempy as scp


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
