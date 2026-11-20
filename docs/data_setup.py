"""Select and verify the documentation's versioned public input data."""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def verify_data():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = DATA / name
        if not path.is_relative_to(DATA) or ".." in path.parts:
            raise ValueError(f"Invalid documentation data path: {name}")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing or changed documentation data: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Changed documentation data: {name}")


def configure_data():
    from spectrochempy import preferences

    preferences.datadir = DATA
