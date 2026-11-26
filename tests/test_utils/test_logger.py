# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import logging
import os
import subprocess
import sys
import warnings
from contextlib import contextmanager

import pytest

from spectrochempy import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    debug_,
    error_,
    get_loglevel,
    info_,
    set_loglevel,
    warning_,
)
from spectrochempy.application.application import app
from spectrochempy.utils import _logging as scp_logging


def test_logger_level_filtering(caplog):
    """Test that log level filtering works correctly for debug_, info_, error_."""
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    set_loglevel(WARNING)
    # debug_ and info_ should still create log records (caplog captures all),
    # but error_ should too
    debug_("debug msg at WARNING level")
    info_("info msg at WARNING level")
    error_(Exception, "error msg at WARNING level")

    assert get_loglevel() == WARNING
    assert any(
        r.levelname == "DEBUG" and "debug msg at WARNING level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "INFO" and "info msg at WARNING level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "ERROR" and "error msg at WARNING level" in r.message
        for r in caplog.records
    )


def test_logger_level_switch(caplog):
    """Test switching log levels and verifying output."""
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    set_loglevel(INFO)
    assert get_loglevel() == INFO

    info_("info msg at INFO level")
    warning_("warning msg at INFO level")
    error_(Exception, "error msg at INFO level")

    assert any(
        r.levelname == "INFO" and "info msg at INFO level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "ERROR" and "error msg at INFO level" in r.message
        for r in caplog.records
    )


def test_warning_function():
    """Test that warning_ emits a warning via the warnings system."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warning_("test warning message for pytest")
        assert len(w) == 1
        assert "test warning message for pytest" in str(w[0].message)


# --------------------------------------------------------------------------------------
# Level applied to the "spectrochempy" loggers (issue #1952)
# --------------------------------------------------------------------------------------
@pytest.fixture
def fresh_scp_logger():
    """Default "spectrochempy" logger without handler; levels restored after."""
    logger = scp_logging._logger
    saved_level = logger.level
    saved_handlers = logger.handlers[:]
    saved_app_level = app.log_level
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    yield logger
    app.log_level = saved_app_level
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)


@contextmanager
def without_root_handlers():
    """Hide pytest's root handlers so that the lazy configuration runs."""
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        yield
    finally:
        root.handlers[:] = saved


def _run_fresh(code, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "SCP_CONFIG_HOME": str(tmp_path), "DOC_BUILDING": "1"},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


_FRESH_PRELUDE = (
    "import sys\n"
    "import spectrochempy as scp\n"
    "from spectrochempy.utils import _logging\n"
    "x = scp.Coord.linspace(1500.0, 1400.0, 201, title='wavenumbers', units='cm^-1')\n"
    "X = scp.gaussian(x, pos=1455.0, width=8.0, normalized=False)\n"
    "def emit(tag):\n"
    "    _logging.info_(f'utils info {tag}')\n"
    "    scp.info_(f'app info {tag}')\n"
    "    _ = X[1600.0:1450.0]\n"
    "    print(f'level {tag} {scp.get_loglevel()}', flush=True)\n"
)


def test_default_hides_info_in_fresh_process(tmp_path):
    """Default level is WARNING: no INFO message is shown (documented behaviour)."""
    out, err = _run_fresh(_FRESH_PRELUDE + "emit('default')\n", tmp_path)
    assert "level default 30" in out
    assert "utils info default" not in err
    assert "app info default" not in err
    assert "outside the axis limits" not in err


def test_set_loglevel_in_fresh_process(tmp_path):
    code = _FRESH_PRELUDE + (
        "scp.set_loglevel('INFO')\n"
        "emit('info')\n"
        "print('---', file=sys.stderr, flush=True)\n"
        "scp.set_loglevel('WARNING')\n"
        "emit('warning')\n"
    )
    out, err = _run_fresh(code, tmp_path)
    shown, hidden = err.split("---")
    assert "level info 20" in out
    assert "utils info info" in shown
    assert "app info info" in shown
    assert "outside the axis limits" in shown
    assert "level warning 30" in out
    assert "utils info warning" not in hidden
    assert "app info warning" not in hidden
    assert "outside the axis limits" not in hidden


def test_config_file_log_level_in_fresh_process(tmp_path):
    """``c.SpectroChemPy.log_level`` in a config file applies to the core logger."""
    (tmp_path / "SpectroChemPy.py").write_text(
        "c = get_config()  # noqa\nc.SpectroChemPy.log_level = 20\n"
    )
    # get_loglevel() starts the application, which reads the config files
    code = _FRESH_PRELUDE + (
        "scp.get_loglevel()\nemit('config')\nprint('logger', _logging._logger.level)\n"
    )
    out, err = _run_fresh(code, tmp_path)
    assert "level config 20" in out
    assert "logger 20" in out
    assert "utils info config" in err
    assert "app info config" in err
    assert "outside the axis limits" in err


def test_app_log_level_applies_to_loggers(fresh_scp_logger):
    for level, expected in (("INFO", INFO), (ERROR, ERROR), ("DEBUG", DEBUG)):
        app.log_level = level
        assert fresh_scp_logger.level == expected
        assert get_loglevel() == expected


def test_model_without_log_level_preserves_both_loggers(tmp_path):
    """A model preserves INFO for application and shared-logger messages."""
    code = _FRESH_PRELUDE + (
        "scp.set_loglevel('INFO')\n"
        "scp.PCA()\n"
        "emit('model')\n"
        "print('logger', _logging._logger.level)\n"
    )
    out, err = _run_fresh(code, tmp_path)
    assert "level model 20" in out
    assert "logger 20" in out
    assert "utils info model" in err
    assert "app info model" in err
    assert "outside the axis limits" in err


def test_lazy_configuration_keeps_default_warning(fresh_scp_logger, capsys):
    with without_root_handlers():
        scp_logging.info_("hidden default info")
        scp_logging.error_("visible error")
        err = capsys.readouterr().err
        assert "hidden default info" not in err
        assert "visible error" in err
        assert fresh_scp_logger.level == logging.WARNING


def test_set_loglevel_before_lazy_configuration(fresh_scp_logger, capsys):
    with without_root_handlers():
        set_loglevel("INFO")
        scp_logging.info_("shown info")
        assert "INFO | shown info" in capsys.readouterr().err
        assert fresh_scp_logger.level == logging.INFO
        assert get_loglevel() == INFO


def test_set_loglevel_after_lazy_configuration(fresh_scp_logger, capsys):
    with without_root_handlers():
        set_loglevel("INFO")
        scp_logging.info_("first info")
        assert "INFO | first info" in capsys.readouterr().err
        set_loglevel(WARNING)
        scp_logging.info_("second info")
        assert "second info" not in capsys.readouterr().err
        set_loglevel("DEBUG")
        scp_logging.debug_("debug message")
        err = capsys.readouterr().err
        assert "DEBUG | debug message" in err
        assert "DEBUG | DEBUG" not in err
        assert get_loglevel() == DEBUG


def test_set_loglevel_notset_shows_all(fresh_scp_logger, capsys):
    with without_root_handlers():
        set_loglevel(0)
        scp_logging.debug_("notset debug")
        assert "DEBUG | notset debug" in capsys.readouterr().err
        assert get_loglevel() == 0


def test_preferences_log_level_applies(fresh_scp_logger, capsys):
    from spectrochempy import preferences

    with without_root_handlers():
        preferences.log_level = INFO
        scp_logging.info_("shown info")
        assert "INFO | shown info" in capsys.readouterr().err
        preferences.log_level = "WARNING"
        scp_logging.info_("hidden info")
        assert "hidden info" not in capsys.readouterr().err
        assert get_loglevel() == WARNING


def test_set_loglevel_applies_to_child_loggers(fresh_scp_logger, caplog):
    caplog.set_level(DEBUG)
    child = logging.getLogger("spectrochempy.some.module")

    set_loglevel("WARNING")
    child.info("child info hidden")
    scp_logging.info_("parent info hidden")
    assert not any("hidden" in r.getMessage() for r in caplog.records)

    set_loglevel("INFO")
    child.info("child info shown")
    scp_logging.info_("parent info shown")
    messages = [r.getMessage() for r in caplog.records]
    assert "child info shown" in messages
    assert "parent info shown" in messages
