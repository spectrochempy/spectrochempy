# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module mainly contains the definition of a Meta class object.

Such object is particularly used in `SpectrochemPy` by the `NDDataset` object
to store metadata. Like a regular dictionary, the
elements can be accessed by key, but also by attributes, *e.g.*
`a = meta['key']` give the same results as `a = meta.key` .
"""

# from traitlets import HasTraits, Dict, Bool, default

# import sys
import copy
import json

import numpy as np

__all__ = []


# ======================================================================================
# Class Meta
# ======================================================================================
class Meta(object):
    """
    A dictionary to store metadata.

    The metadata are accessible by item or by attributes, and
    the dictionary can be made read-only if necessary.

    Parameters
    ----------
    **data : keywords
        The dictionary can be initialized with some keywords.

    Examples
    --------
    First we initialize a metadata object

    >>> m = scp.Meta()

    Then, metadata can be set by attribute (or by key like in a regular
    dictionary), and further accessed by attribute (or key):

    >>> m.chaine = "a string"
    >>> m["entier"] = 123456
    >>> print(m.entier)
    123456
    >>> print(m.chaine)
    a string

    One can make the dictionary read-only

    >>> m.readonly = True
    >>> m.chaine = "a modified string"
    Traceback (most recent call last):
     ...
    ValueError : 'the metadata `chaine` is read only'
    >>> print(m.chaine)
    a string
    """

    # ----------------------------------------------------------------------------------
    # private attributes
    # ----------------------------------------------------------------------------------
    _data = {}  # Internal storage for metadata

    # ----------------------------------------------------------------------------------
    # public attributes
    # ----------------------------------------------------------------------------------
    readonly = False  # Flag to make the dictionary read-only
    parent = None  # Reference to parent object
    name = None  # Name of the metadata object

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    def __init__(self, **data):
        # Initialize the Meta object with optional parent and name attributes.
        self.parent = data.pop("parent", None)
        self.name = data.pop("name", None)
        self._data = data

    def __dir__(self):
        # List of attributes available for the Meta object.
        return ["data", "readonly", "parent", "name"]

    def __setattr__(self, key, value):
        # Set attribute value, ensuring readonly attributes are not modified.
        if key not in [
            "readonly",
            "parent",
            "name",
            "_data",
            "_trait_values",
            "_trait_notifiers",
            "_trait_validators",
            "_cross_validation_lock",
            "__wrapped__",
        ]:
            self[key] = value
        else:
            self.__dict__[key] = value  # Directly set the attribute to avoid recursion

    def __getattr__(self, key):
        # Get attribute value, raising AttributeError for certain keys.
        if key.startswith("_ipython") or key.startswith("_repr"):
            raise AttributeError
        if key in ["__wrapped__"]:
            return False
        return self[key]

    def __setitem__(self, key, value):
        # Set item in the dictionary, respecting readonly flag.
        if key in self.__dir__() or key.startswith("_"):
            raise KeyError("`{}` can not be used as a metadata key".format(key))
        elif not self.readonly:
            self._data.update({key: value})
        else:
            raise ValueError("the metadata `{}` is read only".format(key))

    def __getitem__(self, key):
        # Get item from the dictionary.
        return self._data.get(key, None)

    def __len__(self):
        # Return the number of items in the dictionary.
        return len(self._data)

    def __copy__(self):
        # Create a shallow copy of the Meta object.
        ret = self.__class__()
        ret.update(copy.deepcopy(self._data))
        ret.readonly = self.readonly
        ret.parent = self.parent
        ret.name = self.name
        return ret

    def __deepcopy__(self, memo=None):
        # Create a deep copy of the Meta object.
        return self.__copy__()

    def __eq__(self, other):
        # Check equality with another Meta object or dictionary.
        m1 = self._data
        if hasattr(other, "_data"):
            m2 = other._data
        elif isinstance(other, dict):
            m2 = other
        else:
            return False
        eq = True
        for k, v in m1.items():
            if isinstance(v, list):
                for i, ve in enumerate(v):
                    eq &= np.all(ve == m2[k][i])
            else:
                eq &= np.all(v == m2.get(k, None))
        return eq

    def __ne__(self, other):
        # Check inequality with another Meta object or dictionary.
        return not self.__eq__(other)

    def __iter__(self):
        # Iterate over the keys of the dictionary.
        for item in sorted(self._data.keys()):
            yield item

    def __str__(self):
        # Return string representation of the dictionary.
        return str(self._data)

    def _repr_html_(self):
        # Return HTML representation of the dictionary.
        s = json.dumps(self._data, sort_keys=True, indent=4)
        return s.replace("\n", "<br/>").replace(" ", "&nbsp;")

    # ----------------------------------------------------------------------------------
    # public methods
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _implements(name=None):
        """
        Check if the object implements the Meta class.

        Parameters
        ----------
        name : str, optional
            Name to check against the Meta class.

        Returns
        -------
        bool
            True if the name matches "Meta", False otherwise.
        """
        if name is None:
            return "Meta"
        else:
            return name == "Meta"

    def to_dict(self):
        """
        Transform a metadata dictionary to a regular one.

        Returns
        -------
        dict
            A regular dictionary.
        """
        return self._data

    def get(self, key, default=None):
        """
        Dictionary get method.

        Parameters
        ----------
        key : str
            Key to retrieve.
        default : any, optional
            Default value if the key is not found.

        Returns
        -------
        any
            The value associated with the key, or the default value.
        """
        return self._data.get(key, default)

    def update(self, d):
        """
        Feed a metadata dictionary with the content of another dictionary.

        Parameters
        ----------
        d : dict-like object
            Any dict-like object can be used, such as `dict`, traits `Dict`, or
            another `Meta` object.
        """
        if isinstance(d, Meta) or hasattr(d, "_data"):
            d = d.to_dict()
        if d:
            self._data.update(d)

    def copy(self):
        """
        Return a disconnected copy of self.

        Returns
        -------
        Meta
            A disconnected meta object identical to the original object.
        """
        return self.__copy__()

    def keys(self):
        """
        A list of metadata contained in the object.

        Returns
        -------
        list
            A sorted key's list.

        Examples
        --------
        >>> m = scp.Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> print(m.keys())
        ['si', 'td']

        Alternatively, it is possible to iterate directly on the Meta object

        >>> m = scp.Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> for key in m:
        ...     print(key)
        si
        td
        """
        return [key for key in self]

    def items(self):
        """
        A list of metadata items contained in the object.

        Returns
        -------
        list
            An item list sorted by key.

        Examples
        --------
        >>> m = scp.Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> print(m.items())
        [('si', 20), ('td', 10)]
        """
        return [(key, self[key]) for key in self]

    def swap(self, dim1, dim2, inplace=True):
        """
        Permute meta corresponding to distinct axes to reflect swapping on the
        corresponding data array.

        Parameters
        ----------
        dim1 : int
            First dimension to swap.
        dim2 : int
            Second dimension to swap.
        inplace : bool, optional
            If True, perform the operation in place. Otherwise, return a new Meta object.

        Returns
        -------
        Meta or None
            The new Meta object if inplace is False, otherwise None.
        """
        newmeta = self.copy()

        newmeta.readonly = False
        newmeta.parent = None
        newmeta.name = None

        for key in self:
            if isinstance(self[key], list) and len(self[key]) > 1:
                X = newmeta[key]
                X[dim1], X[dim2] = X[dim2], X[dim1]
            else:
                newmeta[key] = self[key]

        newmeta.readonly = self.readonly
        newmeta.parent = self.parent
        newmeta.name = self.name

        if not inplace:
            return newmeta
        else:
            self._data = newmeta._data

    def permute(self, *dims, inplace=True):
        """
        Permute the metadata according to the given dimensions.

        Parameters
        ----------
        dims : int
            Dimensions to permute.
        inplace : bool, optional
            If True, perform the operation in place. Otherwise, return a new Meta object.

        Returns
        -------
        Meta or None
            The new Meta object if inplace is False, otherwise None.
        """
        newmeta = self.copy()

        newmeta.readonly = False
        newmeta.parent = None
        newmeta.name = None

        for key in self:
            if isinstance(self[key], list) and len(self[key]) > 1:
                newmeta[key] = type(self[key])()
                for dim in dims:
                    newmeta[key].append(self[key][dim])
            else:
                newmeta[key] = self[key]

        newmeta.readonly = self.readonly
        newmeta.parent = self.parent
        newmeta.name = self.name

        if not inplace:
            return newmeta
        else:
            self._data = newmeta._data

    @property
    def data(self):
        """
        Property to access the internal data dictionary.

        Returns
        -------
        dict
            The internal data dictionary.
        """
        return self._data
