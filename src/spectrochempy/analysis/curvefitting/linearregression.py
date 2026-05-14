# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Implementation of least squares Linear Regression."""

import traitlets as tr

from spectrochempy.analysis._base._analysisbase import LinearRegressionAnalysis
from spectrochempy.utils.decorators import signature_has_configurable_traits

__all__ = ["LSTSQ", "NNLS"]
__configurables__ = ["LSTSQ", "NNLS"]


# ======================================================================================
# class LSTSQ
# ======================================================================================
@signature_has_configurable_traits
class LSTSQ(LinearRegressionAnalysis):
    """
    Ordinary least squares Linear Regression (LSTSQ).

    Use :class:`sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients ``w = (w1, ..., wp)``
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        reuse the solution of the previous call to fit and add more components
        (if available) in a sequential manner.

        When `warm_start` is `True`, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`.

    See Also
    --------
    NNLS : Non-Negative least squares Linear Regression.

    """

    name = "LSTSQ"
    description = "Ordinary Least Squares Linear Regression"


# ======================================================================================
# class NNLS
# ======================================================================================
@signature_has_configurable_traits
class NNLS(LinearRegressionAnalysis):
    """
    Non-Negative least squares Linear Regression (NNLS).

    Use :class:`sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    which can not be negative
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        reuse the solution of the previous call to fit and add more components
        (if available) in a sequential manner.

        When `warm_start` is `True`, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`.

    See Also
    --------
    LSTSQ : Ordinary least squares Linear Regression.

    """

    name = "NNLS"
    description = "Non-Negative Least Squares Linear Regression"

    positive = tr.Bool(
        default_value=True,
        help="When set to `True` , forces the coefficients to be positive.",
    ).tag(config=True)
