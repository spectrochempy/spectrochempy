# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
Preferences for writer methods and classes

"""

from traitlets.config.configurable import Configurable

__all__ = []

class WriterPreferences(Configurable):
    """
    Preferences relative to writers and exporters

    """

    def __init__(self, **kwargs):
        super(WriterPreferences, self).__init__(**kwargs)
