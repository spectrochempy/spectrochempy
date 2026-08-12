# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Targeted tests for deprecated plotting method aliases."""

import pytest

from spectrochempy.plotting._methods import normalize_backend_method


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [("stack", "lines"), ("map", "contour")],
)
def test_backend_method_alias_warns_without_promising_013(legacy, canonical):
    warned_aliases = set()

    with pytest.warns(DeprecationWarning, match=rf'method="{legacy}"') as record:
        normalized = normalize_backend_method(legacy, warned_aliases=warned_aliases)

    assert normalized == canonical
    messages = [str(item.message) for item in record]
    assert all("0.13.0" not in message for message in messages)
    assert any(f'method="{canonical}"' in message for message in messages)
    assert any("deprecation policy is satisfied" in message for message in messages)
