# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Various utility objects for SpectroChemPy.

This module provides enhanced dictionary and list classes with additional features
like attribute access and read-only capabilities.
"""

from typing import Any
from typing import TypeVar
from typing import Union

# Type variables for better type hints
T = TypeVar("T")
DictType = Union[dict[str, Any], "Adict", "ReadOnlyDict"]


class Adict(dict):
    """
    A dictionary subclass that allows attribute-style access to its keys.

    This class extends the standard dictionary to allow accessing items using
    dot notation (as attributes) in addition to the standard bracket notation.
    Nested dictionaries are automatically converted to Adict instances.

    Examples
    --------
    >>> d = Adict(a=1, b={'c': 2})
    >>> d.b.c  # Nested access
    2
    >>> d.d = {'e': 3}  # Nested conversion
    >>> d.d.e
    3
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize and convert nested dictionaries."""
        super().__init__()
        if args or kwargs:
            self.update(dict(*args, **kwargs))

    def __getattr__(self, key: str) -> Any:
        """
        Get an attribute, falling back to dictionary key lookup.

        Parameters
        ----------
        key : str
            The attribute/key name to look up

        Returns
        -------
        Any
            The value associated with the key

        Raises
        ------
        AttributeError
            If the key is not found in either __dict__ or the dictionary
        """
        if key in self.__dict__:
            return self.__dict__[key]
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set an attribute, using dictionary storage for non-special attributes.

        Parameters
        ----------
        key : str
            The attribute/key name
        value : Any
            The value to store
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a dictionary item, converting nested dictionaries to Adict.

        Parameters
        ----------
        key : str
            The key to set
        value : Any
            The value to store
        """
        if isinstance(value, dict) and not isinstance(value, Adict):
            value = Adict(value)
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the dictionary with new key/value pairs, converting nested dictionaries.

        Parameters
        ----------
        *args : tuple
            Positional arguments (dictionaries to update from)
        **kwargs : dict
            Keyword arguments to update from
        """
        for k, v in dict(*args, **kwargs).items():
            if isinstance(v, dict):
                if k in self and isinstance(self[k], Adict):
                    self[k].update(v)
                else:
                    self[k] = Adict(v)
            else:
                self[k] = v


class ReadOnlyDict(dict):
    """
    A dictionary that can be made read-only to prevent modifications.

    This class extends dict to add a read-only mode. When enabled,
    attempts to modify the dictionary raise ValueError. It also handles
    nested dictionaries and Meta objects, converting them to ReadOnlyDict
    instances and propagating the read-only state.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to dict constructor
    **kwargs : dict
        Keyword arguments passed to dict constructor

    Examples
    --------
    >>> d = ReadOnlyDict(a=1, b={'c': 2})
    >>> d.readonly = True
    >>> d['a'] = 2  # Raises ValueError
    >>> d['b']['c'] = 3  # Raises ValueError
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with read-only flag disabled and convert nested dicts."""
        self._readonly = False
        super().__init__()
        # Convert the input dictionary and update
        initial_dict = dict(*args, **kwargs)
        converted_dict = {k: self.__convert_nested(v) for k, v in initial_dict.items()}
        super().update(converted_dict)

    def __convert_nested(self, value: Any) -> Any:
        """Convert nested dictionaries to ReadOnlyDict recursively."""
        if isinstance(value, dict) and not isinstance(value, ReadOnlyDict):
            return ReadOnlyDict(value)
        if isinstance(value, list | tuple):
            return type(value)(self.__convert_nested(item) for item in value)
        return value

    def set_readonly(self, readonly: bool) -> None:
        """Set the read-only state recursively for this dictionary and nested objects."""
        self._readonly = readonly
        for value in self.values():
            if isinstance(value, list | tuple):
                # Handle nested dicts in sequences
                for item in value:
                    if isinstance(item, ReadOnlyDict):
                        item.set_readonly(readonly)
            elif isinstance(value, ReadOnlyDict):
                value.set_readonly(readonly)
            elif isinstance(value, dict):
                value = ReadOnlyDict(value)
                value.set_readonly(readonly)
            elif hasattr(value, "_implements") and value._implements("Meta"):
                value.readonly = readonly

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a dictionary item, respecting read-only state."""
        if self._readonly:
            raise ValueError("This dictionary is read-only")
        value = self.__convert_nested(value)
        if hasattr(value, "_implements") and value._implements("Meta"):
            value.readonly = self._readonly
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the dictionary with new key/value pairs, respecting read-only state.

        Parameters
        ----------
        *args : tuple
            Positional arguments (dictionaries to update from)
        **kwargs : dict
            Keyword arguments to update from

        Raises
        ------
        ValueError
            If dictionary is read-only
        """
        if self._readonly:
            raise ValueError("This dictionary is read-only")
        for key, value in dict(*args, **kwargs).items():
            if isinstance(value, dict | Adict):
                if key in self and isinstance(self[key], ReadOnlyDict):
                    self[key].update(value)
                else:
                    self[key] = self.__convert_nested(value)
            elif hasattr(value, "_implements") and value._implements("Meta"):
                if (
                    key in self
                    and hasattr(self[key], "_implements")
                    and self[key]._implements("Meta")
                ):
                    self[key].update(value.to_dict())
                else:
                    self[key] = value
            else:
                self[key] = self.__convert_nested(value)

    def clear(self) -> None:
        """
        Remove all items from the dictionary.

        Raises
        ------
        ValueError
            If dictionary is read-only
        """
        if self._readonly:
            raise ValueError("This dictionary is read-only")
        super().clear()

    def pop(self, key: str, default: Any = None) -> Any:
        """
        Remove specified key and return the corresponding value.

        Parameters
        ----------
        key : str
            Key to remove and return its value
        default : Any, optional
            Value to return if key is not found

        Returns
        -------
        Any
            Value associated with the removed key

        Raises
        ------
        ValueError
            If dictionary is read-only
        KeyError
            If key is not found and no default is provided
        """
        if self._readonly:
            raise ValueError("This dictionary is read-only")
        if default is None:
            return super().pop(key)
        return super().pop(key, default)

    def copy(self) -> "ReadOnlyDict":
        """
        Create a shallow copy of the dictionary.

        Returns
        -------
        ReadOnlyDict
            A new ReadOnlyDict instance with the same content
        """
        new_dict = ReadOnlyDict(self)
        new_dict._readonly = self._readonly
        return new_dict


class ScpObjectList(list):
    """A list subclass that allows html representation of the list of spectrochempy objects."""

    def _repr_html_(self):
        """
        Return the html representation of the list of spectrochempy objects.

        Returns
        -------
        str
            The html representation of the list of spectrochempy objects.
        """
        from spectrochempy.utils.print import convert_to_html

        objtypes = list({item._implements() for item in self})
        objtypes = "mixed" if len(objtypes) > 1 else objtypes[0]
        if objtypes == "_Axes":
            return ""
        html = f"<div class='scp-output'><details><summary>List (len={len(self)}, type={objtypes})</summary><ul>"
        for i, item in enumerate(self):
            html += f"<div class='scp-output section'>{convert_to_html(item, open=False, id=i)}</div>\n"
        html += "</details></div>"
        return html
