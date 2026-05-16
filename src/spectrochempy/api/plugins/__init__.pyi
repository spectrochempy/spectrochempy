# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# ruff: noqa

__all__ = [
    "AnalysisContribution",
    "CORE_PLUGIN_API_VERSION",
    "MissingPluginError",
    "PluginCapability",
    "PluginDescriptor",
    "PluginState",
    "PluginVersionError",
    "ProcessorContribution",
    "ReaderContribution",
    "SimulationContribution",
    "SpectroChemPyPlugin",
    "VisualizerContribution",
    "WriterContribution",
    "analysis_from_dict",
    "check_plugin_compatibility",
    "check_plugin_contributions",
    "check_plugin_metadata",
    "check_plugin_requires",
    "hookimpl",
    "hookspec",
    "processor_from_dict",
    "reader_from_dict",
    "simulation_from_dict",
    "validate_plugin_compatibility",
    "visualizer_from_dict",
    "writer_from_dict",
]

from spectrochempy.api.plugins.base import SpectroChemPyPlugin
from spectrochempy.api.plugins.constants import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins.hooks import hookimpl
from spectrochempy.api.plugins.hooks import hookspec
from spectrochempy.api.plugins.validation import check_plugin_compatibility
from spectrochempy.api.plugins.validation import check_plugin_contributions
from spectrochempy.api.plugins.validation import check_plugin_metadata
from spectrochempy.api.plugins.validation import check_plugin_requires
from spectrochempy.api.plugins.validation import validate_plugin_compatibility
from spectrochempy.plugins.capabilities import PluginCapability
from spectrochempy.plugins.contributions import AnalysisContribution
from spectrochempy.plugins.contributions import ProcessorContribution
from spectrochempy.plugins.contributions import ReaderContribution
from spectrochempy.plugins.contributions import SimulationContribution
from spectrochempy.plugins.contributions import VisualizerContribution
from spectrochempy.plugins.contributions import WriterContribution
from spectrochempy.plugins.contributions import analysis_from_dict
from spectrochempy.plugins.contributions import processor_from_dict
from spectrochempy.plugins.contributions import reader_from_dict
from spectrochempy.plugins.contributions import simulation_from_dict
from spectrochempy.plugins.contributions import visualizer_from_dict
from spectrochempy.plugins.contributions import writer_from_dict
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.deps import PluginVersionError
from spectrochempy.plugins.lifecycle import PluginDescriptor
from spectrochempy.plugins.lifecycle import PluginState
