"""This module is adapted from the brukeropus Python package designed for interacting with Bruker's OPUS spectroscopy software.
It includes only the brukeropus.file module used to read OPUS data files, not the brukeropus.control module used to communicate/control OPUS software
 using the DDE communication protocol."""

# ruff: noqa: F401
from spectrochempy.extern.brukeropus.file import OPUSFile
