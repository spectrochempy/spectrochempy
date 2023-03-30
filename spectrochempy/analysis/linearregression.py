# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of least squares Linear Regression.
"""
import traitlets as tr

from spectrochempy.analysis._base import LinearRegressionAnalysis
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__all__ = ["LSTSQ", "NNLS"]
__configurables__ = ["LSTSQ", "NNLS"]


# ======================================================================================
# class LSTSQ
# ======================================================================================
@signature_has_configurable_traits
class LSTSQ(LinearRegressionAnalysis):
    __doc__ = _docstring.dedent(
        """
    Ordinary least squares Linear Regression (LSTSQ).

    Use :class:`sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients ``w = (w1, ..., wp)``
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    NNLS : Non-Negative least squares Linear Regression.
    """
    )
    name = "LSTSQ"
    description = "Ordinary Least Squares Linear Regression"


# ======================================================================================
# class NNLS
# ======================================================================================
@signature_has_configurable_traits
class NNLS(LinearRegressionAnalysis):
    __doc__ = _docstring.dedent(
        """
    Non-Negative least squares Linear Regression (NNLS).

    Use :class:`sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    which can not be negative
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    NNLS : Ordinary least squares Linear Regression.
    """
    )
    name = "NNLS"
    description = "Non-Negative Least Squares Linear Regression"

    positive = tr.Bool(
        default_value=True,
        help="When set to `True` , forces the coefficients to be positive.",
    ).tag(config=True)
