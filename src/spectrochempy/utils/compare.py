# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np

from .typeutils import is_sequence


def dict_compare(d1, d2, check_equal_only=True):
    """
    Compare two dictionaries.

    Parameters
    ----------
    d1, d2 : dict
        Dictionaries to compare.
    check_equal_only : bool, optional default: True
        If check equal is True, then is the dictionaries are not equal, False is False
        returned. If check is False, for set of values are returned which contains
        added, removed, modified, or same values, respectively.

    Returns
    -------
    results
        Either  bool or a tupl e of for sets if check_equals_only is False.

    Examples
    --------
        >>> from spectrochempy.utils.typeutils import dict_compare
        >>>
        >>> x = dict(a=1, b=2)
        >>> y = dict(a=2, b=2)
        >>> added, removed, modified, same = dict_compare(x, y, check_equal_only=False)
        >>> print(added, removed, modified, same)
        set() set() {'a'} {'b'}
        >>> dict_compare(x, y)
        False

    """
    # from
    # http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
    # modified to account for the comparison of list objects

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys

    modified = added.union(removed)
    for o in intersect_keys:
        if is_sequence(d1[o]):
            if not is_sequence(d2[o]) or len(d1[o]) != len(d2[o]):
                modified.add(o)
            else:
                # in principe we vae here two list of same length
                for i1, i2 in zip(d1[o], d2[o], strict=False):
                    if np.any(i1 != i2):
                        modified.add(o)
        elif is_sequence(d2[o]) or d1[o] != d2[o]:
            modified.add(o)

    same = {o for o in intersect_keys if o not in modified}

    if not check_equal_only:
        return added, removed, modified, same
    return not (modified or removed or added)


def difference(x, y):
    """
    Calculate absolute and relative differences between two datasets.

    Parameters
    ----------
    x, y : NDDataset
        Datasets to compare

    Returns
    -------
    tuple
        Maximum absolute error and maximum relative error
    """
    from numpy import abs
    from numpy import max

    nonzero = (x.data != 0) & (y.data != 0)
    error = abs(x.data - y.data)
    ax = abs(x.data)[nonzero]
    ay = abs(y.data)[nonzero]
    max_relative_error = max((error[nonzero] / ax, error[nonzero] / ay))
    return max(error), max_relative_error
