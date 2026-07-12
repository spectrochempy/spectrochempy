# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests verifying NMR/Hypercomplex decoupling (no private imports)."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.plugin

_HYPERCOMPLEX_AVAILABLE = (
    importlib.util.find_spec("spectrochempy_hypercomplex") is not None
)
_NMR_AVAILABLE = importlib.util.find_spec("spectrochempy_nmr") is not None


@pytest.mark.skipif(
    not _HYPERCOMPLEX_AVAILABLE, reason="spectrochempy-hypercomplex not installed"
)
def test_hypercomplex_public_api_is_exported():
    """spectrochempy_hypercomplex must expose quaternion utilities publicly."""
    from spectrochempy_hypercomplex import as_float_array
    from spectrochempy_hypercomplex import as_quat_array
    from spectrochempy_hypercomplex import as_quaternion
    from spectrochempy_hypercomplex import is_available

    assert callable(as_float_array)
    assert callable(as_quat_array)
    assert callable(as_quaternion)
    assert isinstance(is_available, bool)


@pytest.mark.skipif(not _NMR_AVAILABLE, reason="spectrochempy-nmr not installed")
def test_nmr_fft_encodings_use_public_hypercomplex_imports():
    """NMR fft_encodings must not reach into hypercomplex private _quaternion."""
    from spectrochempy_nmr.processing import fft_encodings

    source = inspect.getsource(fft_encodings)
    assert "spectrochempy_hypercomplex._quaternion" not in source
    assert "from quaternion import" not in source
    assert "from spectrochempy_hypercomplex import" in source


@pytest.mark.skipif(not _NMR_AVAILABLE, reason="spectrochempy-nmr not installed")
def test_nmr_read_topspin_uses_public_hypercomplex_imports():
    """read_topspin must not import as_quat_array directly from numpy-quaternion."""
    import spectrochempy_nmr.readers.read_topspin as rt_mod

    source = inspect.getsource(rt_mod)
    assert "from quaternion import as_quat_array" not in source
    assert "from spectrochempy_hypercomplex import as_quat_array" in source


@pytest.mark.skipif(not _NMR_AVAILABLE, reason="spectrochempy-nmr not installed")
@pytest.mark.skipif(
    not _HYPERCOMPLEX_AVAILABLE, reason="spectrochempy-hypercomplex not installed"
)
def test_fft_encodings_graceful_when_hypercomplex_absent(monkeypatch):
    """Hypercomplex encodings raise a clear ImportError when hypercomplex is missing."""
    from spectrochempy_nmr.processing.fft_encodings import _states_fft

    def _raise(*args, **kwargs):
        raise ImportError("simulated hypercomplex absence")

    monkeypatch.setattr("spectrochempy_hypercomplex.as_float_array", _raise)
    monkeypatch.setattr("spectrochempy_hypercomplex.as_quaternion", _raise)

    # Create dummy quaternion-like data (numpy-quaternion is available in test env)
    from quaternion import as_quat_array

    data = as_quat_array(np.zeros((4, 4, 4)))

    with pytest.raises(ImportError):
        _states_fft(data)
