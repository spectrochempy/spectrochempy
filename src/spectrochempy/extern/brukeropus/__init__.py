"""Vendored file reader adapted from Josh Duran's ``brukeropus`` package.

The vendored reader is synchronized with upstream commit
``af5a508cef7de8089acd27a215d644ab451257dd`` (2025-11-14). SpectroChemPy
retains the file-like input support and internal import paths required by its
reader adapter, and does not depend on ``brukeropus`` at runtime.

Only the ``brukeropus.file`` module is included; the upstream DDE control
module is intentionally excluded.
"""

# lazy_stub: skip
# ruff: noqa: F401
from spectrochempy.extern.brukeropus.file import OPUSFile
