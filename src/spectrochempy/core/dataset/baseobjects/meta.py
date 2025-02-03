# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Meta Class Module.

Provides metadata storage for SpectroChemPy objects through the Meta class.
Metadata can be accessed both as dictionary items and as attributes.

Features:
- Dictionary-like access (meta['key']) and attribute access (meta.key)
- Read-only protection option
- JSON serialization support
- Deep copy support

Classes
-------
Meta
    Dictionary-like metadata container with attribute access
"""

__all__ = []

import copy
import json
from typing import Any
from typing import Optional
from typing import Union

import numpy as np


class Meta:
    """
    Dictionary-like metadata container with attribute access.

    Provides storage for metadata that can be accessed either through dictionary
    keys or object attributes. Can be made read-only to prevent modifications.

    Parameters
    ----------
    **data : Any
        Initial metadata as keyword arguments

    Examples
    --------
    Create metadata and set values:

    >>> meta = Meta()
    >>> meta.title = "My Dataset"
    >>> meta["author"] = "Me"
    >>> print(meta.title)
    My Dataset

    Make read-only:

    >>> meta.readonly = True
    >>> meta.title = "Changed"  # Raises ValueError

    """

    _data: dict[str, Any] = {}
    readonly: bool = False
    parent: Any | None = None
    name: str | None = None

    def __init__(self, **data: Any) -> None:
        # Initialize metadata object
        self.parent = data.pop("parent", None)
        self.name = data.pop("name", None)
        self._data = data

    def __dir__(self) -> list[str]:
        # List available attributes
        return ["data", "readonly", "parent", "name"]

    def __setattr__(self, key: str, value: Any) -> None:
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

    def __getattr__(self, key: str) -> Any:
        # Get attribute value, raising AttributeError for certain keys.
        if key.startswith("_ipython") or key.startswith("_repr"):
            raise AttributeError
        if key in ["__wrapped__"]:
            return False
        return self[key]

    def __setitem__(self, key: str, value: Any) -> None:
        # Set item in the dictionary, respecting readonly flag.
        if key in self.__dir__() or key.startswith("_"):
            raise KeyError(f"`{key}` can not be used as a metadata key")
        if not self.readonly:
            self._data.update({key: value})
        else:
            raise ValueError(f"the metadata `{key}` is read only")

    def __getitem__(self, key: str) -> Any:
        # Get item from the dictionary.
        return self._data.get(key, None)

    def __len__(self) -> int:
        # Return the number of items in the dictionary.
        return len(self._data)

    def __copy__(self) -> "Meta":
        # Create a shallow copy of the Meta object.
        ret = self.__class__()
        ret.update(copy.deepcopy(self._data))
        ret.readonly = self.readonly
        ret.parent = self.parent
        ret.name = self.name
        return ret

    def __deepcopy__(self, memo=None) -> "Meta":
        # Create a deep copy of the Meta object.
        return self.__copy__()

    def __eq__(self, other: object) -> bool:
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

    def __ne__(self, other: object) -> bool:
        # Check inequality with another Meta object or dictionary.
        return not self.__eq__(other)

    def __iter__(self):
        # Iterate over the keys of the dictionary.
        yield from sorted(self._data.keys())

    def __str__(self) -> str:
        # Return string representation of the dictionary.
        return str(self._data)

    def _repr_html_(self) -> str:
        # Return HTML representation of the dictionary.
        s = json.dumps(self._data, sort_keys=True, indent=4)
        return s.replace("\n", "<br/>").replace(" ", "&nbsp;")

    @staticmethod
    def _implements(name: str | None = None) -> str | bool:
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
        return name == "Meta"

    def to_dict(self) -> dict:
        """
        Convert metadata to regular dictionary.

        Returns
        -------
        dict
            Dictionary containing the metadata

        """
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value by key.

        Parameters
        ----------
        key : str
            Key to retrieve
        default : Any, optional
            Default value if key not found

        Returns
        -------
        Any
            Value for key or default

        """
        return self._data.get(key, default)

    def update(self, d: Union[dict, "Meta"]) -> None:
        """
        Update metadata from dictionary or Meta object.

        Parameters
        ----------
        d : Union[Dict, Meta]
            Source of metadata to update from

        """
        if isinstance(d, Meta) or hasattr(d, "_data"):
            d = d.to_dict()
        if d:
            self._data.update(d)

    def copy(self) -> "Meta":
        """
        Create a shallow copy.

        Returns
        -------
        Meta
            New Meta instance with copied data

        """
        return self.__copy__()

    def keys(self) -> list[str]:
        """
        Get sorted list of metadata keys.

        Returns
        -------
        List[str]
            Sorted list of metadata keys

        """
        return list(self)

    def items(self) -> list[tuple[str, Any]]:
        """
        Get sorted list of metadata items.

        Returns
        -------
        List[tuple[str, Any]]
            List of (key, value) tuples sorted by key

        """
        return [(key, self[key]) for key in self]

    def swap(self, dim1: int, dim2: int, inplace: bool = True) -> Optional["Meta"]:
        """
        Swap metadata between dimensions.

        Parameters
        ----------
        dim1 : int
            First dimension to swap
        dim2 : int
            Second dimension to swap
        inplace : bool, optional
            Whether to modify in place

        Returns
        -------
        Optional[Meta]
            New Meta object if not inplace

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
        self._data = newmeta._data
        return None

    def permute(self, *dims: int, inplace: bool = True) -> Optional["Meta"]:
        """
        Permute metadata dimensions.

        Parameters
        ----------
        *dims : int
            Dimensions to permute to
        inplace : bool, optional
            Whether to modify in place

        Returns
        -------
        Optional[Meta]
            New Meta object if not inplace

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
        self._data = newmeta._data
        return None

    @property
    def data(self) -> dict[str, Any]:
        """
        Access internal data dictionary.

        Returns
        -------
        Dict[str, Any]
            The metadata dictionary

        """
        return self._data
