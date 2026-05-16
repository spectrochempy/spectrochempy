# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Plugin capability enumeration.

Provides a shared vocabulary for plugin capabilities so that
different plugins use consistent capability names.
"""

from __future__ import annotations

from enum import Enum


class PluginCapability(Enum):
    """
    Standardised plugin capability names.

    Every plugin should declare its capabilities using these values so
    that the plugin-manager and the registry can validate and normalise
    capability strings automatically.

    Example::

        from spectrochempy.plugins.capabilities import PluginCapability

        class MyPlugin:
            capabilities = [PluginCapability.READER, PluginCapability.WRITER]
    """

    READER = "reader"
    WRITER = "writer"
    PROCESSOR = "processor"
    VISUALIZER = "visualizer"
    ANALYSIS = "analysis"
    SIMULATION = "simulation"
    ACCESSOR = "accessor"
