# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Configuration persistence helpers for SpectroChemPy."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from traitlets.config.manager import BaseJSONConfigManager
from traitlets.config.manager import recursive_update

from spectrochempy.utils._logging import warning_


class SpectroChemPyJSONConfigManager(BaseJSONConfigManager):
    """
    JSON config manager with in-process locking and atomic writes.

    The shared per-file ``RLock`` protects the whole logical update sequence
    (read -> merge -> serialize -> replace) for threads inside one Python
    process. Atomic replacement prevents readers from observing partially
    written JSON files.

    This does not coordinate separate Python processes writing the same file.
    """

    _file_locks_guard = threading.Lock()
    _file_locks: dict[str, threading.RLock] = {}

    @classmethod
    def _lock_for_path(cls, path: Path) -> threading.RLock:
        key = os.fspath(path)
        with cls._file_locks_guard:
            lock = cls._file_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._file_locks[key] = lock
        return lock

    def _section_path(self, section_name: str) -> Path:
        return Path(self.file_name(section_name))

    def _warn_corrupt_file(
        self,
        *,
        filename: Path,
        backup: Path,
        exc: json.JSONDecodeError,
    ) -> None:
        warning_(
            "Recovered corrupted SpectroChemPy config JSON "
            f"{filename!s} by moving it to {backup!s}: {exc}",
            UserWarning,
        )

    def _read_json_file(self, filename: Path) -> dict:
        with filename.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _atomic_write_json(self, filename: Path, data: dict) -> None:
        self.ensure_config_dir_exists()

        tmp_name = None
        file_mode = None
        with suppress(FileNotFoundError):
            file_mode = stat.S_IMODE(filename.stat().st_mode)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=filename.parent,
                prefix=f".{filename.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

            if file_mode is not None:
                os.chmod(tmp_name, file_mode)

            os.replace(tmp_name, filename)
        except Exception:
            if tmp_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(tmp_name)
            raise

    def _next_corrupt_backup_path(self, filename: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = filename.with_name(f"{filename.name}.corrupt-{timestamp}")
        counter = 1
        while backup.exists():
            backup = filename.with_name(f"{filename.name}.corrupt-{timestamp}-{counter}")
            counter += 1
        return backup

    def _recover_corrupted_file(
        self,
        filename: Path,
        exc: json.JSONDecodeError,
    ) -> dict:
        backup = self._next_corrupt_backup_path(filename)
        os.replace(filename, backup)
        self._warn_corrupt_file(filename=filename, backup=backup, exc=exc)
        self._atomic_write_json(filename, {})
        return {}

    def _read_section_data_unlocked(self, filename: Path) -> dict:
        if not filename.is_file():
            return {}
        try:
            return self._read_json_file(filename)
        except json.JSONDecodeError as exc:
            return self._recover_corrupted_file(filename, exc)

    def ensure_file_is_valid(self, filename: str | Path) -> bool:
        path = Path(filename)
        if not path.exists():
            return True

        with self._lock_for_path(path):
            try:
                self._read_json_file(path)
            except json.JSONDecodeError as exc:
                self._recover_corrupted_file(path, exc)
                return False
        return True

    def get(self, section_name: str):
        filename = self._section_path(section_name)
        with self._lock_for_path(filename):
            return self._read_section_data_unlocked(filename)

    def set(self, section_name: str, data):
        filename = self._section_path(section_name)
        with self._lock_for_path(filename):
            self._atomic_write_json(filename, data)

    def update(self, section_name: str, new_data):
        filename = self._section_path(section_name)
        with self._lock_for_path(filename):
            data = self._read_section_data_unlocked(filename)
            recursive_update(data, new_data)
            self._atomic_write_json(filename, data)
            return data
