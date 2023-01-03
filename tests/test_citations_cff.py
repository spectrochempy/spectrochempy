# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

from spectrochempy.utils.citation import Zenodo, Citation


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
