import logging
from importlib import import_module
from io import StringIO
from unittest.mock import Mock

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.application import app

optimize_module = import_module("spectrochempy.analysis.curvefitting.optimize")


@pytest.fixture(autouse=True)
def preserve_log_level():
    level = scp.get_loglevel()
    yield
    scp.set_loglevel(level)


@pytest.fixture
def spectrum():
    x = np.linspace(-3, 3, 31)
    return scp.NDDataset(np.exp(-(x**2)), coordset=[scp.Coord(x)])


def _optimizer(method="least_squares", log_level="WARNING", dry=False):
    model = scp.Optimize(
        method=method,
        log_level=log_level,
        dry=dry,
        callback_every=1,
        max_iter=8,
    )
    model.script = """
MODEL: LINE_1
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos: 0.0, -2.0, 2.0
    $ width: 1.0, 0.1, 5.0
"""
    return model


@pytest.mark.parametrize("notebook", [False, True])
@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING"])
@pytest.mark.parametrize("method", ["least_squares", "simplex"])
@pytest.mark.parametrize("dry", [False, True])
def test_output_clearing_requires_notebook_and_info(
    spectrum, monkeypatch, notebook, level, method, dry
):
    clear = Mock()
    monkeypatch.setattr(optimize_module.display, "clear_output", clear)
    monkeypatch.setattr(optimize_module, "is_notebook", lambda: notebook)
    model = _optimizer(method, level, dry)
    console_output = StringIO()
    monkeypatch.setattr(app.log.handlers[0], "stream", console_output)

    model.fit(spectrum)

    verbose = level in ("DEBUG", "INFO")
    assert clear.called == (notebook and verbose)
    assert all(call.kwargs == {"wait": True} for call in clear.call_args_list)
    messages = console_output.getvalue()
    summary = "Starting parameters:" if dry else "Result:"
    assert (summary in messages) == verbose
    progress = "Iterations:" in messages
    assert progress == (verbose and method == "simplex" and not dry)


@pytest.mark.parametrize("level", ["INFO", "WARNING"])
def test_terminal_fit_emits_no_control_characters(spectrum, monkeypatch, capsys, level):
    monkeypatch.setattr(optimize_module, "is_notebook", lambda: False)
    model = _optimizer(log_level=level)
    capsys.readouterr()

    model.fit(spectrum)

    assert "\x1b" not in capsys.readouterr().out


@pytest.mark.parametrize("level", ["INFO", "WARNING"])
def test_output_clearing_uses_current_log_level(spectrum, monkeypatch, level):
    clear = Mock()
    monkeypatch.setattr(optimize_module.display, "clear_output", clear)
    monkeypatch.setattr(optimize_module, "is_notebook", lambda: True)
    model = _optimizer(log_level="WARNING" if level == "INFO" else "INFO")
    scp.set_loglevel(level)

    model.fit(spectrum)

    assert clear.called == (level == "INFO")


def test_quiet_fit_preserves_warnings_and_errors(spectrum, monkeypatch):
    monkeypatch.setattr(optimize_module, "is_notebook", lambda: True)
    clear = Mock()
    monkeypatch.setattr(optimize_module.display, "clear_output", clear)
    model = _optimizer(method="simplex")
    console_output = StringIO()
    monkeypatch.setattr(app.log.handlers[0], "stream", console_output)

    with pytest.warns(UserWarning, match="Maximum number of iterations reached"):
        model.fit(spectrum)
    scp.error_("Error message remains visible")

    clear.assert_not_called()
    assert scp.get_loglevel() == logging.WARNING
    assert "Error message remains visible" in console_output.getvalue()
