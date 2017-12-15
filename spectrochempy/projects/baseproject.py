# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
# See full LICENSE agreement in the root directory
# ============================================================================


import uuid

from traitlets import HasTraits

__all__ = []


class AbstractProject(HasTraits):
    """
    Project class will subclass it. This is mainly for type comparison purpose
    """
