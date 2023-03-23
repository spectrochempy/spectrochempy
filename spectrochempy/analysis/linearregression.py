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

from spectrochempy.analysis._base import (
    LinearRegressionAnalysis,
    _make_other_parameters_doc,
)
from spectrochempy.utils.docstrings import _docstring

__all__ = ["LSTSQ", "NNLS"]
__configurables__ = ["LSTSQ", "NNLS"]


# ======================================================================================
# class LSTSQ
# ======================================================================================
class LSTSQ(LinearRegressionAnalysis):
    __doc__ = _docstring.dedent(
        """
    Ordinary least squares Linear Regression.

    Use :class:`~sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    Other Parameters
    ----------------
    {{CONFIGURATION_PARAMETERS}}

    See Also
    --------
    NNLS : Non-Negative least squares Linear Regression.
    """
    )
    name = "LSTSQ"
    description = "Ordinary Least Squares Linear Regression"


_make_other_parameters_doc(LSTSQ)


# ======================================================================================
# class NNLS
# ======================================================================================
class NNLS(LinearRegressionAnalysis):
    __doc__ = _docstring.dedent(
        """
    Non-Negative least squares Linear Regression.

    Use :class:`~sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    which can not be negative
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    Other Parameters
    ----------------
    {{CONFIGURATION_PARAMETERS}}

    See Also
    --------
    NNLS : Ordinary least squares Linear Regression.
    """
    )
    name = "NNLS"
    description = "Non-Negative Least Squares Linear Regression"

    positive = tr.Bool(
        default_value=True,
        help="When set to ``True``, forces the coefficients to be positive. This"
        "option is only supported for dense arrays.",
    ).tag(config=True)


_make_other_parameters_doc(NNLS)
