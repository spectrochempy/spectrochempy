# import pytest

from spectrochempy.utils.citation import Zenodo, Citation


def test_zenodo_update():
    zenodo = Zenodo()
    zenodo.load()
    zenodo.update_version()
    zenodo.update_date()
    zenodo.save()

    assert str(zenodo) is not None


def test_citation_update():
    citation = Citation()
    citation.load()
    citation.update_version()
    citation.update_date()
    citation.save()

    assert str(citation) is not None
    assert citation.apa == str(citation)
