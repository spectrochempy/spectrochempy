import pytest

# import spectrochempy
from spectrochempy.core.analysis import cantera_utilities as cu


@pytest.mark.skipif(
    cu._cantera_is_not_available(), reason="Cantera must be installed first"
)
def test_cu(monkeypatch):

    # availability of cantera (# should be installed if the test wa not skipped)
    assert not cu._cantera_is_not_available()

    # simulate abscense of cantera
    with monkeypatch.context() as m:
        m.setattr(cu, "ct", None)
        assert cu._cantera_is_not_available()

    # context restored with ct = cantera
    assert not cu._cantera_is_not_available()


# TODO: tests someexamples
