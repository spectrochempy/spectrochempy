#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import io

import spectrochempy as scp


def test_show_versions() -> None:
    f = io.StringIO()
    scp.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
    assert "python:" in f.getvalue()
