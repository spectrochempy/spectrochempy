# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for robust JSON config persistence."""

from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path

from spectrochempy.application.config import SpectroChemPyJSONConfigManager


def test_update_serializes_whole_read_modify_write_sequence(tmp_path):
    """Distinct concurrent updates should both survive in the final JSON."""
    manager = SpectroChemPyJSONConfigManager(config_dir=str(tmp_path))
    entered_first_write = threading.Event()
    release_first_write = threading.Event()
    first_call = True
    original_atomic_write = manager._atomic_write_json

    def controlled_atomic_write(filename, data):
        nonlocal first_call
        if first_call:
            first_call = False
            entered_first_write.set()
            assert release_first_write.wait(timeout=5)
        original_atomic_write(filename, data)

    manager._atomic_write_json = controlled_atomic_write

    def update_alpha():
        manager.update("Shared", {"Shared": {"alpha": 1}})

    def update_beta():
        manager.update("Shared", {"Shared": {"beta": 2}})

    thread_alpha = threading.Thread(target=update_alpha)
    thread_beta = threading.Thread(target=update_beta)

    thread_alpha.start()
    assert entered_first_write.wait(timeout=5)
    thread_beta.start()
    release_first_write.set()

    thread_alpha.join(timeout=5)
    thread_beta.join(timeout=5)

    path = tmp_path / "Shared.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"Shared": {"alpha": 1, "beta": 2}}


def test_update_recovers_corrupted_json_without_stdout(tmp_path, capsys):
    """A broken JSON file should be backed up, warned about and recreated."""
    manager = SpectroChemPyJSONConfigManager(config_dir=str(tmp_path))
    path = tmp_path / "Shared.json"
    path.write_text('{"Shared": {"alpha": 1}}\n}\n', encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.update("Shared", {"Shared": {"beta": 2}})
        manager.update("Shared", {"Shared": {"gamma": 3}})

    backups = sorted(tmp_path.glob("Shared.json.corrupt-*"))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert capsys.readouterr().out == ""
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8").endswith("}\n}\n")
    assert payload == {"Shared": {"beta": 2, "gamma": 3}}
    assert len(caught) == 1
    assert "Recovered corrupted SpectroChemPy config JSON" in str(caught[0].message)
    assert str(path) in str(caught[0].message)


def test_ensure_file_is_valid_repairs_existing_invalid_json(tmp_path):
    """Startup validation should preserve a corrupted file and recreate JSON."""
    manager = SpectroChemPyJSONConfigManager(config_dir=str(tmp_path))
    path = tmp_path / "GeneralPreferences.json"
    path.write_text("{not valid json", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        valid_before = manager.ensure_file_is_valid(path)

    backups = sorted(tmp_path.glob("GeneralPreferences.json.corrupt-*"))

    assert valid_before is False
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"
    assert json.loads(path.read_text(encoding="utf-8")) == {}
    assert len(caught) == 1


def test_atomic_write_replaces_shorter_json_without_trailing_fragment(tmp_path):
    """Replacing a longer document with a shorter one should stay exact JSON."""
    manager = SpectroChemPyJSONConfigManager(config_dir=str(tmp_path))
    path = Path(manager.file_name("Shared"))

    manager.set("Shared", {"Shared": {"alpha": [1, 2, 3, 4], "beta": "long"}})
    manager.set("Shared", {"Shared": {"beta": "short"}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"Shared": {"beta": "short"}}
