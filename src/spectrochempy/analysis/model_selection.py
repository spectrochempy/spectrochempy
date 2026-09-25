# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Cross-validation splitters exposed by SpectroChemPy."""

from sklearn.model_selection import GroupKFold as _SklearnGroupKFold
from sklearn.model_selection import KFold as _SklearnKFold
from sklearn.model_selection import LeaveOneOut as _SklearnLeaveOneOut

__all__ = ["GroupKFold", "KFold", "LeaveOneOut"]


class KFold(_SklearnKFold):
    """
    Split observations into consecutive cross-validation folds.

    This is a thin SpectroChemPy adaptation of
    :class:`sklearn.model_selection.KFold`. It uses scikit-learn's partitioning
    algorithm unchanged while providing documentation and an explicit public
    signature for use with :func:`spectrochempy.cross_validate`.

    Parameters
    ----------
    n_splits : int, optional, default: 5
        Number of folds. Must be at least 2 and no greater than the number of
        observations.
    shuffle : bool, optional, default: False
        Shuffle observation positions before dividing them into folds. The
        observations within each resulting fold are not shuffled.
    random_state : int, RandomState instance or None, optional, default: None
        Controls the ordering when *shuffle* is true. Pass an integer to obtain
        reproducible folds. It has no effect when *shuffle* is false, and
        scikit-learn rejects a non-None value in that case.

    See Also
    --------
    cross_validate : Execute supervised cross-validation.
    GroupKFold : Keep groups separated between folds.
    LeaveOneOut : Validate one observation at a time.
    sklearn.model_selection.KFold : Underlying implementation.

    Notes
    -----
    The splitter produces integer positions. ``scp.cross_validate`` resolves
    ``sample_dim``, validates coordinates, slices the ``NDDataset`` inputs, and
    fits each fold. Calling :meth:`split` directly does not make the splitter
    interpret named dimensions or coordinates automatically.

    Examples
    --------
    >>> X = scp.NDDataset([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    >>> y = scp.NDDataset([[0.0], [1.0], [2.0], [3.0]])
    >>> model = scp.PLSRegression(n_components=1)
    >>> splitter = scp.KFold(n_splits=2, shuffle=True, random_state=7)
    >>> result = scp.cross_validate(model, X, y, cv=splitter)
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )


class GroupKFold(_SklearnGroupKFold):
    """
    Split observations while keeping each group in a single validation fold.

    This is a thin SpectroChemPy adaptation of
    :class:`sklearn.model_selection.GroupKFold`. It uses scikit-learn's
    partitioning algorithm unchanged. Each distinct group appears in exactly
    one validation fold and is never shared between the calibration and
    validation subsets of a fold.

    Parameters
    ----------
    n_splits : int, optional, default: 5
        Number of folds. Must be at least 2 and no greater than the number of
        distinct groups.

    See Also
    --------
    cross_validate : Execute supervised cross-validation.
    KFold : Split observations without group constraints.
    sklearn.model_selection.GroupKFold : Underlying implementation.

    Notes
    -----
    The public signature intentionally contains only parameters supported
    across the scikit-learn versions used by SpectroChemPy. Group assignment is
    deterministic for a fixed input order.

    The splitter produces integer positions. ``scp.cross_validate`` resolves
    ``sample_dim``, validates coordinates and group identities, slices the
    ``NDDataset`` inputs, and fits each fold. Calling :meth:`split` directly
    does not make the splitter interpret named dimensions or coordinates
    automatically.

    Examples
    --------
    >>> X = scp.NDDataset([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    >>> y = scp.NDDataset([[0.0], [1.0], [2.0], [3.0]])
    >>> model = scp.PLSRegression(n_components=1)
    >>> sample_groups = [0, 0, 1, 1]
    >>> splitter = scp.GroupKFold(n_splits=2)
    >>> result = scp.cross_validate(
    ...     model, X, y, cv=splitter, groups=sample_groups
    ... )
    """

    def __init__(self, n_splits=5):
        super().__init__(n_splits=n_splits)


class LeaveOneOut(_SklearnLeaveOneOut):
    """
    Use each observation once as a one-observation validation fold.

    This is a thin SpectroChemPy adaptation of
    :class:`sklearn.model_selection.LeaveOneOut`. It uses scikit-learn's
    partitioning algorithm unchanged and creates as many fits as there are
    observations.

    See Also
    --------
    cross_validate : Execute supervised cross-validation.
    KFold : Select a fixed number of folds.
    sklearn.model_selection.LeaveOneOut : Underlying implementation.

    Notes
    -----
    Leave-one-out validation can be expensive because it fits the estimator
    once per observation. R² is undefined for each one-observation fold;
    ``scp.cross_validate`` records that limitation while its global out-of-fold
    R² may still be defined when enough valid observations are available.

    The splitter produces integer positions. ``scp.cross_validate`` resolves
    ``sample_dim``, validates coordinates, slices the ``NDDataset`` inputs, and
    fits each fold. Calling :meth:`split` directly does not make the splitter
    interpret named dimensions or coordinates automatically.

    Examples
    --------
    >>> X = scp.NDDataset([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    >>> y = scp.NDDataset([[0.0], [1.0], [2.0], [3.0]])
    >>> model = scp.PLSRegression(n_components=1)
    >>> splitter = scp.LeaveOneOut()
    >>> result = scp.cross_validate(model, X, y, cv=splitter)
    """

    def __init__(self):
        super().__init__()
