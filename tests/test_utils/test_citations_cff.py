# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("cffconvert", reason="cffconvert not installed") is None,
    reason="cffconvert not installed",
)


from spectrochempy.utils.citation import Citation, Zenodo


def test_zenodo_update():
    zenodo = Zenodo()
    zenodo.load()
    zenodo.update_version()
    zenodo.update_date()
    assert str(zenodo) is not None


def test_citation_update():
    citation = Citation()
    citation.load()
    citation.update_version()
    citation.update_date()
    assert str(citation) is not None
    assert citation.apa == str(citation)
