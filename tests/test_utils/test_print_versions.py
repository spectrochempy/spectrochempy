# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import io

import spectrochempy as scp


def test_show_versions() -> None:
    f = io.StringIO()
    scp.show_versions(file=f)
    assert "INSTALLED PACKAGES" in f.getvalue()
    assert "python" in f.getvalue()
    assert "numpy" in f.getvalue()
    assert "pint" in f.getvalue()
    assert "matplotlib" in f.getvalue()
    assert "pytest" in f.getvalue()
    assert "SPECTROCHEMPY" in f.getvalue()


if __name__ == "__main__":
    test_show_versions()
