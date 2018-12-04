# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ============================================================================

__all__ = []

import uuid

from traitlets import HasTraits




class AbstractProject(HasTraits):
    """
    Project class will subclass it. This is mainly for type comparison purpose
    """

#TODO: replace this by an implement attributes in such classes