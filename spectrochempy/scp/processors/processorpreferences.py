# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
Preferences for processor methods and classes

"""
from traitlets.config.configurable import Configurable

__all__ = []

class ProcessorPreferences(Configurable):
    """
    Preferences relative to processing

    """
    def __init__(self, **kwargs):
        super(ProcessorPreferences, self).__init__(**kwargs)

