# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""
"""
__all__ = ['get_user_and_node',
           'get_user',
           'get_node',
           'is_kernel',
           ]

import getpass
import platform
import inspect
import os
import sys


def get_user():
    return getpass.getuser()


def get_node():
    return platform.node()


def get_user_and_node():
    return "{0}@{1}".format(get_user(), get_node())


def is_kernel():
    """ Check if we are running from IPython

    """
    # from http://stackoverflow.com
    # /questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None

