# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

__all__ = ["NBlack", "NRed", "NBlue", "NGreen"]


# color conversion function
def _cmyk2rgb(C, M, Y, K):
    """
    CMYK to RGB conversion.

    C,M,Y,K are given in percent.
    The R,G,B values are returned in the range of 0..1.
    """
    C, Y, M, K = C / 100.0, Y / 100.0, M / 100.0, K / 100.0

    # The red (R) color is calculated from the cyan (C) and black (K) colors:
    R = (1.0 - C) * (1.0 - K)

    # The green color (G) is calculated from the magenta (M) and black (K) colors:
    G = (1.0 - M) * (1.0 - K)

    # The blue color (B) is calculated from the yellow (Y) and black (K) colors:
    B = (1.0 - Y) * (1.0 - K)

    return R, G, B


# colors
# --------------------------------------------------------------------------------------
# For color blind people, it is safe to use only 4 colors in graphs:
# see http://jfly.iam.u-tokyo.ac.jp/color/ichihara_etal_2008.pdf
#   Black CMYK=0,0,0,0
#   Red CMYK= 0, 77, 100, 0 %
#   Blue CMYK= 100, 30, 0, 0 %
#   Green CMYK= 85, 0, 60, 10 %
NBlack = (0, 0, 0)
NRed = _cmyk2rgb(0, 77, 100, 0)
NBlue = _cmyk2rgb(100, 30, 0, 0)
NGreen = _cmyk2rgb(85, 0, 60, 10)
# TODO : make a color cycle based on these colors
