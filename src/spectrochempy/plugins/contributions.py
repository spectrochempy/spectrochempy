# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Lightweight contribution data models for plugin registrations.

These dataclasses provide typed structure for plugin contributions
while coexisting with the existing loose-dict format.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ReaderContribution:
    """
    Describes a file-reader contribution from a plugin.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"topspin"``).
    func : Callable
        The reader callable.
    description : str
        Human-readable description.
    extensions : list[str] or None
        File extensions handled by this reader (e.g. ``[".fid"]``).
    """

    name: str
    func: Callable
    description: str = ""
    extensions: list[str] | None = None


@dataclass
class WriterContribution:
    """
    Describes a file-writer contribution from a plugin.

    Parameters
    ----------
    name : str
        Short identifier.
    func : Callable
        The writer callable.
    description : str
        Human-readable description.
    """

    name: str
    func: Callable
    description: str = ""


@dataclass
class VisualizerContribution:
    """
    Describes a visualizer contribution from a plugin.

    Parameters
    ----------
    name : str
        Short identifier.
    func : Callable
        The visualizer callable.
    description : str
        Human-readable description.
    """

    name: str
    func: Callable
    description: str = ""


@dataclass
class ProcessorContribution:
    """
    Describes a data-processor contribution from a plugin.

    Parameters
    ----------
    name : str
        Short identifier.
    func : Callable
        The processor callable.
    description : str
        Human-readable description.
    """

    name: str
    func: Callable
    description: str = ""


# ------------------------------------------------------------------
# Conversion helpers (dict ↔ dataclass)
# ------------------------------------------------------------------


def reader_from_dict(d: dict[str, Any]) -> ReaderContribution:
    """Convert a loosely-typed dict to a ``ReaderContribution``."""
    return ReaderContribution(
        name=d["name"],
        func=d["func"],
        description=d.get("description", ""),
        extensions=d.get("extensions"),
    )


def writer_from_dict(d: dict[str, Any]) -> WriterContribution:
    """Convert a loosely-typed dict to a ``WriterContribution``."""
    return WriterContribution(
        name=d["name"],
        func=d["func"],
        description=d.get("description", ""),
    )


def visualizer_from_dict(d: dict[str, Any]) -> VisualizerContribution:
    """Convert a loosely-typed dict to a ``VisualizerContribution``."""
    return VisualizerContribution(
        name=d["name"],
        func=d["func"],
        description=d.get("description", ""),
    )


def processor_from_dict(d: dict[str, Any]) -> ProcessorContribution:
    """Convert a loosely-typed dict to a ``ProcessorContribution``."""
    return ProcessorContribution(
        name=d["name"],
        func=d["func"],
        description=d.get("description", ""),
    )
