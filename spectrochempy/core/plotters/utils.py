# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import numpy as np


__all__ = []
_methods = ['make_label', 'cmyk2rgb']


def make_label(ss, lab='<no_axe_label>'):
    """ make a label from title and units

    """
    if ss.title:
        label = ss.title  # .replace(' ', r'\ ')
    else:
        label = lab

    if ss.units is not None and str(ss.units) != 'dimensionless':
        #if str(ss.units) == 'absorbance':
        #    units = r'/\ a.u.'
        #else:
        units = r"/\ {:~L}".format(ss.units)
        units = units.replace('%',r'\%')
    else:
        units = ''

    label = r"%s $\mathrm{%s}$" % (label, units)
    return label


# color conversion function
def cmyk2rgb(C, M, Y, K):
    """CMYK to RGB conversion
    C,M,Y,K are given in percent.
    The R,G,B values are returned in the range of 0..1.
    """
    C, Y, M, K = C / 100., Y / 100., M / 100., K / 100.

    # The red (R) color is calculated from the cyan (C) and black (K) colors:
    R = (1.0 - C) * (1.0 - K)

    # The green color (G) is calculated from the magenta (M) and black (K) colors:
    G = (1. - M) * (1. - K)

    # The blue color (B) is calculated from the yellow (Y) and black (K) colors:
    B = (1. - Y) * (1. - K)

    return (R, G, B)




def make_attr(key):
    name = 'M_%s' % key[1]
    k = r'$\mathrm{%s}$' % name

    if 'P' in name:
        m = 'o'
        c = NBlack
    elif 'A' in name:
        m = '^'
        c = NBlue
    elif 'B' in name:
        m = 's'
        c = NRed

    if '400' in key:
        f = 'w'
        s = ':'
    else:
        f = c
        s = '-'

    return m, c, k, f, s
