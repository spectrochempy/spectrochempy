# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
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
