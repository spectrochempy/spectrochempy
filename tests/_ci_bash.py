"""Locate the shell used by workflow regression tests."""

import shutil
import sys
from pathlib import Path


def resolve_bash():
    if sys.platform == "win32":
        git = shutil.which("git")
        if git:
            for directory in Path(git).resolve().parents[:3]:
                bash = directory / "bin" / "bash.exe"
                if bash.is_file():
                    return str(bash)
        return None
    return shutil.which("bash")
