"""Model construction respects the shared logging configuration."""

import logging

import pytest

import spectrochempy as scp
from spectrochempy.application.application import app
from spectrochempy.utils.baseconfigurable import BaseConfigurable

MODELS = [
    BaseConfigurable,
    scp.Baseline,
    scp.PCA,
    scp.NMF,
    scp.EFA,
    scp.SIMPLISMA,
    scp.FastICA,
    scp.SVD,
    scp.MCRALS,
    scp.PLSRegression,
    scp.Optimize,
    scp.Filter,
    scp.PSD,
    scp.LSTSQ,
    scp.NNLS,
]


def assert_log_level(expected):
    assert scp.get_loglevel() == expected
    assert app.log_level == expected
    assert logging.getLogger("spectrochempy").getEffectiveLevel() == expected


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
)
@pytest.mark.parametrize("kwargs", [{}, {"log_level": None}], ids=["omitted", "none"])
def test_constructor_preserves_global_level(model, level, kwargs):
    scp.set_loglevel(level)
    model(**kwargs)
    assert_log_level(level)


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
@pytest.mark.parametrize(
    "level, expected", [("WARNING", logging.WARNING), (logging.INFO, logging.INFO)]
)
def test_constructor_explicit_level_changes_global(model, level, expected):
    scp.set_loglevel("DEBUG")
    model(log_level=level)
    assert_log_level(expected)
    scp.Baseline()
    scp.PCA()
    assert_log_level(expected)
