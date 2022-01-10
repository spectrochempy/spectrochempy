# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements class |CoordSet|.
"""

__all__ = ["CoordSet"]

import copy as cpy
import warnings
import uuid

import numpy as np
from traitlets import (
    HasTraits,
    List,
    Bool,
    Unicode,
    observe,
    All,
    validate,
    default,
    Dict,
    Int,
)

from spectrochempy.core.dataset.ndarray import NDArray, DEFAULT_DIM_NAME
from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.utils import (
    is_sequence,
    colored_output,
    convert_to_html,
    SpectroChemPyWarning,
)


# ======================================================================================================================
# CoordSet
# ======================================================================================================================
class CoordSet(HasTraits):
    # Hidden attributes containing the collection of objects
    _coords = List(allow_none=True)
    _references = Dict()
    _updated = Bool(False)

    # Hidden id and name of the object
    _id = Unicode()
    _name = Unicode()

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = Bool(False)

    # other settings
    _copy = Bool(False)
    _sorted = Bool(True)
    _html_output = Bool(False)

    # default coord index
    _default = Int(0)

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    # ..........................................................................
    def __init__(self, *coords, **kwargs):
        """
        A collection of Coord objects for a NDArray object with validation.

        This object is an iterable containing a collection of Coord objects.

        Parameters
        ----------
        *coords : |NDarray|, |NDArray| subclass or |CoordSet| sequence of objects.
            If an instance of CoordSet is found, instead of an array, this means
            that all coordinates in this coords describe the same axis.
            It is assumed that the coordinates are passed in the order of the
            dimensions of a nD numpy array (
            `row-major <https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
            order), i.e., for a 3d object : 'z', 'y', 'x'.
        **kwargs: dict
            See other parameters.

        Other Parameters
        ----------------
        x : |NDarray|, |NDArray| subclass or |CoordSet|
            A single coordinate associated to the 'x'-dimension.
            If a coord was already passed in the argument, this will overwrite
            the previous. It is thus not recommended to simultaneously use
            both way to initialize the coordinates to avoid such conflicts.
        y, z, u, ... : |NDarray|, |NDArray| subclass or |CoordSet|
            Same as `x` for the others dimensions.
        dims : list of string, optional
            Names of the dims to use corresponding to the coordinates. If not given, standard names are used: x, y, ...
        copy : bool, optional
            Perform a copy of the passed object. Default is True.

        See Also
        --------
        Coord : Explicit coordinates object.
        LinearCoord : Implicit coordinates object.
        NDDataset: The main object of SpectroChempy which makes use of CoordSet.

        Examples
        --------
        >>> from spectrochempy import Coord, CoordSet

        Define 4 coordinates, with two for the same dimension

        >>> coord0 = Coord.linspace(10., 100., 5, units='m', title='distance')
        >>> coord1 = Coord.linspace(20., 25., 4, units='K', title='temperature')
        >>> coord1b = Coord.linspace(1., 10., 4, units='millitesla', title='magnetic field')
        >>> coord2 = Coord.linspace(0., 1000., 6, units='hour', title='elapsed time')

        Now create a coordset

        >>> cs = CoordSet(t=coord0, u=coord2, v=[coord1, coord1b])

        Display some coordinates

        >>> cs.u
        Coord: [float64] hr (size: 6)

        >>> cs.v
        CoordSet: [_1:temperature, _2:magnetic field]

        >>> cs.v_1
        Coord: [float64] K (size: 4)
        """

        self._copy = kwargs.pop("copy", True)
        self._sorted = kwargs.pop("sorted", True)

        keepnames = kwargs.pop("keepnames", False)
        # if keepnames is false and the names of the dimensions are not passed in kwargs, then use dims if not none
        dims = kwargs.pop("dims", None)

        self.name = kwargs.pop("name", None)

        # initialise the coordinate list
        self._coords = []

        # First evaluate passed args
        # --------------------------

        # some cleaning
        if coords:

            if all(
                [
                    (
                        isinstance(coords[i], (np.ndarray, NDArray, list, CoordSet))
                        or coords[i] is None
                    )
                    for i in range(len(coords))
                ]
            ):
                # Any instance of a NDArray can be accepted as coordinates for a dimension.
                # If an instance of CoordSet is found, this means that all
                # coordinates in this set describe the same axis
                coords = tuple(coords)

            elif is_sequence(coords) and len(coords) == 1:
                # if isinstance(coords[0], list):
                #     coords = (CoordSet(*coords[0], sorted=False),)
                # else:
                coords = coords[0]

                if isinstance(coords, dict):
                    # we have passed a dict, postpone to the kwargs evaluation process
                    kwargs.update(coords)
                    coords = None

            else:
                raise ValueError("Did not understand the inputs")

        # now store the args coordinates in self._coords (validation is fired when this attribute is set)
        if coords:
            for coord in coords[::-1]:  # we fill from the end of the list
                # (in reverse order) because by convention when the
                # names are not specified, the order of the
                # coords follow the order of dims.
                if not isinstance(coord, CoordSet):
                    if isinstance(coord, list):
                        coord = CoordSet(*coord[::-1], sorted=False)
                    elif not isinstance(coord, LinearCoord):  # else
                        coord = Coord(coord, copy=True)
                else:
                    coord = cpy.deepcopy(coord)

                if not keepnames:
                    if dims is None:
                        # take the last available name of available names list
                        coord.name = self.available_names.pop(-1)
                    else:
                        # use the provided list of dims
                        coord.name = dims.pop(-1)

                self._append(coord)  # append the coord (but instead of append,
                # use assignation -in _append - to fire the validation process )

        # now evaluate keywords argument
        # ------------------------------

        for key, coord in list(kwargs.items())[:]:
            # remove the already used kwargs (Fix: deprecation warning in Traitlets - all args, kwargs must be used)
            del kwargs[key]

            # prepare values to be either Coord, LinearCoord or CoordSet
            if isinstance(coord, (list, tuple)):
                coord = CoordSet(
                    *coord, sorted=False
                )  # make sure in this case it becomes a CoordSet instance

            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(
                    coord, copy=True
                )  # make sure it's a Coord  # (even if it is None -> Coord(None)

            elif isinstance(coord, str) and coord in DEFAULT_DIM_NAME:
                # may be a reference to another coordinates (e.g. same coordinates for various dimensions)
                self._references[key] = coord  # store this reference
                continue

            # Populate the coords with coord and coord's name.
            if isinstance(coord, (NDArray, Coord, LinearCoord, CoordSet)):  # NDArray,
                if key in self.available_names or (
                    len(key) == 2
                    and key.startswith("_")
                    and key[1] in list("123456789")
                ):
                    # ok we can find it as a canonical name:
                    # this will overwrite any already defined coord value
                    # which means also that kwargs have priority over args
                    coord.name = key
                    self._append(coord)

                elif not self.is_empty and key in self.names:
                    # append when a coordinate with this name is already set in passed arg.
                    # replace it
                    idx = self.names.index(key)
                    coord.name = key
                    self._coords[idx] = coord

                else:
                    raise KeyError(
                        f"Probably an invalid key (`{key}`) for coordinates has been passed. "
                        f"Valid keys are among:{DEFAULT_DIM_NAME}"
                    )

            else:
                raise ValueError(
                    f"Probably an invalid type of coordinates has been passed: {key}:{coord} "
                )

        # store the item (validation will be performed)
        # self._coords = _coords

        # inform the parent about the update
        self._updated = True

        # set a notifier on the name traits name of each coordinates
        for coord in self._coords:
            if coord is not None:
                HasTraits.observe(coord, self._coords_update, "_name")

        # initialize the base class with the eventual remaining arguments
        super().__init__(**kwargs)

    # ..........................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `CoordSet`.

        Rather than isinstance(obj, CoordSet) use object.implements('CoordSet').

        This is useful to check type without importing the module.
        """
        if name is None:
            return "CoordSet"
        else:
            return name == "CoordSet"

    # ------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    @validate("_coords")
    def _coords_validate(self, proposal):
        coords = proposal["value"]
        if not coords:
            return None

        for id, coord in enumerate(coords):
            if coord and not isinstance(coord, (Coord, LinearCoord, CoordSet)):
                raise TypeError(
                    "At this point all passed coordinates should be of type Coord or CoordSet!"
                )  # coord =  #
                # Coord(coord)
            if self._copy:
                coords[id] = coord.copy()
            else:
                coords[id] = coord

        for coord in coords:
            if isinstance(coord, CoordSet):
                # it must be a single dimension axis
                # in this case we must have same length for all coordinates
                coord._is_same_dim = True

                # check this is valid in term of size
                try:
                    coord.sizes
                except ValueError:
                    raise

                # change the internal names
                n = len(coord)
                coord._set_names(
                    [f"_{i + 1}" for i in range(n)]
                )  # we must have  _1 for the first coordinates,
                # _2 the second, etc...
                coord._set_parent_dim(coord.name)

        # last check and sorting
        names = []
        for coord in coords:
            if coord.has_defined_name:
                names.append(coord.name)
            else:
                raise ValueError(
                    "At this point all passed coordinates should have a valid name!"
                )

        if coords:
            if self._sorted:
                _sortedtuples = sorted(
                    (coord.name, coord) for coord in coords
                )  # Final sort
                coords = list(zip(*_sortedtuples))[1]
            return list(coords)  # be sure its a list not a tuple
        else:
            return None

    # ..........................................................................
    @default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------

    # ..........................................................................
    @property
    def available_names(self):
        """
        Chars that can be used for dimension name (list).

        It returns DEFAULT_DIM_NAMES less those already in use.
        """
        _available_names = DEFAULT_DIM_NAME.copy()
        for item in self.names:
            if item in _available_names:
                _available_names.remove(item)
        return _available_names

    # ..........................................................................
    @property
    def coords(self):
        """
        Coordinates in the coordset (list).
        """
        return self._coords

    # ..........................................................................
    @property
    def has_defined_name(self):
        """
        True if the name has been defined (bool).
        """
        return not (self.name == self.id)

    # ..........................................................................
    @property
    def id(self):
        """
        Object identifier (Readonly property).
        """
        return self._id

    # ..........................................................................
    @property
    def is_empty(self):
        """
        True if there is no coords defined (bool).
        """
        if self._coords:
            return len(self._coords) == 0
        else:
            return False

    # ..........................................................................
    @property
    def is_same_dim(self):
        """
        True if the coords define a single dimension (bool).
        """
        return self._is_same_dim

    # ..........................................................................
    @property
    def references(self):
        return self._references

    # ..........................................................................
    @property
    def sizes(self):
        """
        Sizes of the coord object for each dimension (int or tuple of int).

        (readonly property). If the set is for a single dimension return a
        single size as all coordinates must have the same.
        """
        _sizes = []
        for i, item in enumerate(self._coords):
            _sizes.append(item.size)  # recurrence if item is a CoordSet

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise ValueError(
                    "Coordinates must be of the same size for a dimension with multiple coordinates"
                )
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    # ..........................................................................
    # @property
    # def coords(self):  #TODO: replace with itertiems, items etc ... to simulate a dict
    #     """
    #     list - list of the Coord objects in the current coords (readonly
    #     property).
    #     """
    #     return self._coords

    # ..........................................................................
    @property
    def names(self):
        """
        Names of the coords in the current coords (list - read only property).
        """
        _names = []
        if self._coords:
            for item in self._coords:
                if item.has_defined_name:
                    _names.append(item.name)
        return _names

    # ------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------

    @property
    def default(self):
        """
        Default coordinates (Coord).
        """
        return self[self._default]

    @property
    def data(self):
        # in case data is called on a coordset for dimension with multiple coordinates
        # return the first coordinates
        return self.default.data

    # ..........................................................................
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self._id

    @name.setter
    def name(self, value):
        if value is not None:
            self._name = value

    # ..........................................................................
    @property
    def titles(self):
        """
        Titles of the coords in the current coords (list).
        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)  # TODO:name
            elif isinstance(item, CoordSet):
                _titles.append(
                    [el.title if el.title else el.name for el in item]
                )  # TODO:name
            else:
                raise ValueError("Something wrong with the titles!")
        return _titles

    # ..........................................................................
    @property
    def labels(self):
        """
        Labels of the coordinates in the current coordset (list).
        """
        return [item.labels for item in self]

    # ..........................................................................
    @property
    def units(self):
        """
        Units of the coords in the current coords (list).
        """
        return [item.units for item in self]

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    def copy(self, keepname=False):
        """
        Make a disconnected copy of the current coords.

        Returns
        -------
        object
            an exact copy of the current object
        """
        return self.__copy__()

    # ..........................................................................
    def keys(self):
        """
        Alias for names.

        Returns
        -------
        out : list
            list of all coordinates names (including reference to other coordinates).
        """
        keys = []
        if self.names:
            keys.extend(self.names)
        if self._references:
            keys.extend(list(self.references.keys()))
        return keys

    # ..........................................................................
    def select(self, val):
        """
        Select the default coord index.
        """
        self._default = min(max(0, int(val) - 1), len(self.names))

    # ...........................................................................................................
    def set(self, *args, **kwargs):
        """
        Set one or more coordinates in the current CoordSet.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        """
        if not args and not kwargs:
            return

        if len(args) == 1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]

        if isinstance(args, CoordSet):
            kwargs.update(args.to_dict())
            args = ()

        if args:
            self._coords = []  # reset

        for i, item in enumerate(args[::-1]):
            item.name = self.available_names.pop()
            self._append(item)

        for k, item in kwargs.items():
            if isinstance(item, CoordSet):
                # try to keep this parameter to True!
                item._is_same_dim = True
            self[k] = item

    # ..........................................................................
    def set_titles(self, *args, **kwargs):
        """
        Set one or more coord title at once.

        Parameters
        ----------
        args : str(s)
            The list of titles to apply to the set of coordinates (they must be given according to the coordinate's name
            alphabetical order.
        **kwargs : str
            Keyword attribution of the titles. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set titles as this will be less prone to errors.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's  name alhabetical order :
        e.g, the first title will be for the `x` coordinates, the second for the `y`, etc.
        """
        if len(args) == 1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self[i], CoordSet):
                self[i].title = item
            else:
                if is_sequence(item):
                    for j, v in enumerate(self[i]):
                        v.title = item[j]

        for k, item in kwargs.items():
            self[k].title = item

    # ..........................................................................
    def set_units(self, *args, **kwargs):
        """
        Set one or more coord units at once.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's name alhabetical order :
        e.g, the first units will be for the `x` coordinates, the second for the `y`, etc.

        Parameters
        ----------
        *args : str(s)
            The list of units to apply to the set of coordinates (they must be given according to the coordinate's name
            alphabetical order.
        **kwargs : str
            Keyword attribution of the units. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set units as this will be less prone to errors.
        force : bool, optional, default=False
            Whether or not the new units must be compatible with the current units. See the `Coord`.`to` method.
        """
        force = kwargs.pop("force", False)

        if len(args) == 1 and is_sequence(args[0]):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self[i], CoordSet):
                self[i].to(item, force=force, inplace=True)
            else:
                if is_sequence(item):
                    for j, v in enumerate(self[i]):
                        v.to(item[j], force=force, inplace=True)

        for k, item in kwargs.items():
            self[k].to(item, force=force, inplace=True)

    # ..........................................................................
    def to_dict(self):
        """
        Return a dict of the coordinates from the coordset.

        Returns
        -------
        out : dict
            A dictionary where keys are the names of the coordinates, and the values the coordinates themselves.
        """
        return dict(zip(self.names, self._coords))

    # ..........................................................................
    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        k**warg
            Only keywords among the CoordSet.names are allowed - they denotes the name of a dimension.
        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self[idx] = Coord(kwargs.pop(dim), name=dim)

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def _append(self, coord):
        # utility function to append coordinate with full validation
        if not isinstance(coord, tuple):
            coord = (coord,)
        if self._coords:
            # some coordinates already present, prepend the new one
            self._coords = (*coord,) + tuple(
                self._coords
            )  # instead of append, fire the validation process
        else:
            # no coordinates yet, start a new tuple of coordinate
            self._coords = (*coord,)

    # ..........................................................................
    def _loc2index(self, loc):
        # Return the index of a location
        for coord in self.coords:
            try:
                return coord._loc2index(loc)
            except IndexError:
                continue
        # not found!
        raise IndexError

    # ..........................................................................
    def _set_names(self, names):
        # utility function to change names of coordinates (in batch)
        # useful when a coordinate is a CoordSet itself
        for coord, name in zip(self._coords, names):
            coord.name = name

    # ..........................................................................
    def _set_parent_dim(self, name):
        # utility function to set the paretn name for sub coordset
        for coord in self._coords:
            coord._parent_dim = name

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    # @staticmethod
    def __dir__(self):
        return ["coords", "references", "is_same_dim", "name"]

    # ..........................................................................
    def __call__(self, *args, **kwargs):
        # allow the following syntax: coords(), coords(0,2) or
        coords = []
        axis = kwargs.get("axis", None)
        if args:
            for idx in args:
                coords.append(self[idx])
        elif axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            for i in axis:
                coords.append(self[i])
        else:
            coords = self._coords
        if len(coords) == 1:
            return coords[0]
        else:
            return CoordSet(*coords)

    # ..........................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    # ..........................................................................
    def __len__(self):
        return len(self._coords)

    def __delattr__(self, item):
        if "notify_change" in item:
            pass

        else:
            try:
                return self.__delitem__(item)
            except (IndexError, KeyError):
                raise AttributeError

    # ..........................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if "_validate" in item or "_changed" in item:
            raise AttributeError

        try:
            return self.__getitem__(item)
        except (IndexError, KeyError):
            raise AttributeError

    # ..........................................................................
    def __getitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                return self._coords.__getitem__(idx)

            # ok we did not find it!
            # let's try in references
            if index in self._references.keys():
                return self._references[index]

            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                        f" the first occurence is returned!"
                    )
                return self._coords.__getitem__(self.titles.index(index))

            # may be it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    return item.__getitem__(item.titles.index(index))

            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord name
                    return item.__getitem__(item.names.index(index))

            try:
                # let try with the canonical dimension names
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        if isinstance(c, CoordSet):
                            c = c.__getitem__(index[1:])
                        else:
                            c = c.__getitem__(index[2:])  # try on labels
                    return c
            except IndexError:
                pass

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

        try:
            self._coords.__getitem__(index)
        except TypeError:
            print()
        res = self._coords.__getitem__(index)
        if isinstance(index, slice):
            if isinstance(res, CoordSet):
                res = (res,)
            return CoordSet(*res, keepnames=True)
        else:
            return res

    # ..........................................................................
    def __setattr__(self, key, value):
        keyb = key[1:] if key.startswith("_") else key
        if keyb in [
            "parent",
            "copy",
            "sorted",
            "coords",
            "updated",
            "name",
            "html_output",
            "is_same_dim",
            "parent_dim",
            "trait_values",
            "trait_notifiers",
            "trait_validators",
            "cross_validation_lock",
            "notify_change",
        ]:
            super().__setattr__(key, value)
            return

        try:
            self.__setitem__(key, value)
        except Exception:
            super().__setattr__(key, value)

    # ..........................................................................
    def __setitem__(self, index, coord):
        try:
            coord = coord.copy(keepname=True)  # to avoid modifying the original
        except TypeError as e:
            if isinstance(coord, list):
                coord = [c.copy(keepname=True) for c in coord[:]]
            else:
                raise e

        if isinstance(index, str):
            # find by name
            if index in self.names:
                idx = self.names.index(index)
                coord.name = index
                self._coords.__setitem__(idx, coord)
                return

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                        f" the first occurence is returned!",
                        SpectroChemPyWarning,
                    )
                index = self.titles.index(index)
                coord.name = self.names[index]
                self._coords.__setitem__(index, coord)
                return

            # may be it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    index = item.titles.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord title
                    index = item.names.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return

            try:
                # let try with the canonical dimension names
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        c.__setitem__(index[1:], coord)
                    return

            except KeyError:
                pass

            # add the new coordinates
            if index in self.available_names or (
                len(index) == 2
                and index.startswith("_")
                and index[1] in list("123456789")
            ):
                coord.name = index
                self._coords.append(coord)
                return

            else:
                raise KeyError(
                    f"Could not find `{index}` in coordinates names or titles"
                )

        self._coords[index] = coord

    # ..........................................................................
    def __delitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                del self._coords[idx]
                return

            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                index = self.titles.index(index)
                self._coords.__delitem__(index)
                return

            # may be it is a title in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    return item.__delitem__(index)

            # let try with the canonical dimension names
            if index[0] in self.names:
                # ok we can find it a a canonical name:
                c = self._coords.__getitem__(self.names.index(index[0]))
                if len(index) > 1 and index[1] == "_":
                    if isinstance(c, CoordSet):
                        return c.__delitem__(index[1:])

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    # ..........................................................................
    # def __iter__(self):
    #    for item in self._coords:
    #        yield item

    # ..........................................................................
    def __repr__(self):
        out = "CoordSet: [" + ", ".join(["{}"] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace("CoordSet: ", ""))
            else:
                s.append(f"{item.name}:{item.title}")
        out = out.format(*s)
        return out

    # ..........................................................................
    def __str__(self):
        return repr(self)

    # ..........................................................................
    def _cstr(self, header="  coordinates: ... \n", print_size=True):

        txt = ""
        for idx, dim in enumerate(self.names):
            coord = getattr(self, dim)

            if coord:

                dimension = f"     DIMENSION `{dim}`"
                for k, v in self.references.items():
                    if dim == v:
                        # reference to this dimension
                        dimension += f"=`{k}`"
                txt += dimension + "\n"

                if isinstance(coord, CoordSet):
                    # txt += '        index: {}\n'.format(idx)
                    if not coord.is_empty:
                        if print_size:
                            txt += f"{coord[0]._str_shape().rstrip()}\n"

                        coord._html_output = self._html_output
                        for idx_s, dim_s in enumerate(coord.names):
                            c = getattr(coord, dim_s)
                            txt += f"          ({dim_s}) ...\n"
                            c._html_output = self._html_output
                            sub = c._cstr(
                                header="  coordinates: ... \n", print_size=False
                            )  # , indent=4, first_indent=-6)
                            txt += f"{sub}\n"

                elif not coord.is_empty:
                    # coordinates if available
                    # txt += '        index: {}\n'.format(idx)
                    coord._html_output = self._html_output
                    txt += "{}\n".format(
                        coord._cstr(header=header, print_size=print_size)
                    )

        txt = txt.rstrip()  # remove the trailing '\n'

        if not self._html_output:
            return colored_output(txt.rstrip())
        else:
            return txt.rstrip()

    # ..........................................................................
    def _repr_html_(self):
        return convert_to_html(self)

    # ..........................................................................
    def __deepcopy__(self, memo):
        coords = self.__class__(
            tuple(cpy.deepcopy(ax, memo=memo) for ax in self), keepnames=True
        )
        coords.name = self.name
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        return coords

    # ..........................................................................
    def __copy__(self):
        coords = self.__class__(tuple(cpy.copy(ax) for ax in self), keepnames=True)
        # name must be changed
        coords.name = self.name
        # and is_same_dim and default for coordset
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        return coords

        # ..........................................................................

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self._coords == other._coords
        except Exception:
            return False

    # ..........................................................................
    def __ne__(self, other):
        return not self.__eq__(other)

    # ------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------
    # ..........................................................................
    def _coords_update(self, change):
        # when notified that a coord name have been updated
        self._updated = True

    # ..........................................................................
    @observe(All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change.name == "_updated" and change.new:
            self._updated = False  # reset


# ======================================================================================================================
if __name__ == "__main__":
    pass
