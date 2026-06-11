# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module that implements class `CoordSet` ."""

__all__ = ["CoordSet"]

import copy as cpy
import uuid
import warnings
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from traitlets import All
from traitlets import Bool
from traitlets import Dict
from traitlets import HasTraits
from traitlets import Int
from traitlets import List
from traitlets import Unicode
from traitlets import default
from traitlets import observe
from traitlets import signature_has_traits
from traitlets import validate

from spectrochempy.core.dataset._coordgroup import _CoordinateEntry
from spectrochempy.core.dataset._coordgroup import _coordset_to_groups
from spectrochempy.core.dataset._coordgroup import _DimensionCoordinates
from spectrochempy.core.dataset._coordgroup import _groups_to_coordset
from spectrochempy.core.dataset._coordgroup import _make_entry_id
from spectrochempy.core.dataset.basearrays.ndarray import DEFAULT_DIM_NAME
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.print import colored_output
from spectrochempy.utils.print import convert_to_html
from spectrochempy.utils.typeutils import is_sequence


# ======================================================================================
# CoordSet
# ======================================================================================
@dataclass(frozen=True)
class _CoordLookupResult:
    """Private lookup result carrying legacy value plus resolver context."""

    value: object
    lookup_kind: str
    owner_dim: str | None = None
    compat_key: str | int | None = None
    warning_message: str | None = None
    entry_id: str | None = None
    alias: str | None = None
    reference_target: str | None = None


@signature_has_traits
class CoordSet(HasTraits):
    r"""
    A collection of Coord objects for a NDArray object with validation.

    This object is an iterable containing a collection of Coord objects.

    Parameters
    ----------
    *coords : `NDArray` ,  `NDArray` subclass or `CoordSet` sequence of objects.
        If an instance of CoordSet is found, instead of an array, this means
        that all coordinates in this coords describe the same axis.
        It is assumed that the coordinates are passed in the order of the
        dimensions of a nD numpy array (
        `row-major <https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
        order), i.e., for a 3d object : 'z', 'y', 'x'.
    **kwargs
        Additional keyword parameters (see Other Parameters).

    Other Parameters
    ----------------
    x : `NDArray` ,  `NDArray` subclass or `CoordSet`
        A single coordinate associated to the 'x'-dimension.
        If a coord was already passed in the argument, this will overwrite
        the previous. It is thus not recommended to simultaneously use
        both way to initialize the coordinates to avoid such conflicts.
    y, z, u, ... : `NDArray` ,  `NDArray` subclass or `CoordSet`
        Same as `x` for the others dimensions.
    dims : list of string, optional
        Names of the dims to use corresponding to the coordinates. If not
        given, standard names are used: x, y, ...
    copy : bool, optional
        Perform a copy of the passed object. Default is True.

    See Also
    --------
    Coord : Explicit coordinates object.
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

    # Hidden attributes containing the collection of objects
    _coords = List()
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

    # ----------------------------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, *coords, **kwargs):
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
                (
                    isinstance(coords[i], np.ndarray | NDArray | list | CoordSet)
                    or coords[i] is None
                )
                for i in range(len(coords))
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
                    else:
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

            # prepare values to be either Coord or CoordSet
            if isinstance(coord, list | tuple):
                coord = CoordSet(
                    *coord,
                    sorted=False,
                )  # make sure in this case it becomes a CoordSet instance

            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(
                    coord,
                    copy=True,
                )  # make sure it's a Coord  # (even if it is None -> Coord(None)

            elif isinstance(coord, str) and coord in DEFAULT_DIM_NAME:
                # may be a reference to another coordinates (e.g. same coordinates for
                # various dimensions)
                self._references[key] = coord  # store this reference
                continue

            # Populate the coords with coord and coord's name.
            if isinstance(coord, NDArray | Coord | CoordSet):
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
                    # append when a coordinate with this name is already set in passed
                    # arg.
                    # replace it
                    idx = self.names.index(key)
                    coord.name = key
                    self._coords[idx] = coord

                else:
                    raise KeyError(
                        f"Probably an invalid key (`{key}` ) for coordinates has been passed. "
                        f"Valid keys are among:{DEFAULT_DIM_NAME}",
                    )

            else:
                raise ValueError(
                    f"Probably an invalid type of coordinates has been passed: {key}:{coord} ",
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

    @staticmethod
    def _implements(name=None):
        """
        Check if the current object implements `CoordSet`.

        Rather than isinstance(obj, CoordSet) use object._implements('CoordSet').

        This is useful to check type without importing the module.
        """
        if name is None:
            return "CoordSet"
        return name == "CoordSet"

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------

    def __sub__(self, other):
        """
        Subtraction of Coordsets.

        Parameters
        ----------
        other : CoordSet
            The Coordset to subtract to self.

        Returns
        -------
        sub : CoordSet
            The difference of the Coordsets.

        """
        out = []
        if isinstance(other, CoordSet):
            for coord1, coord2 in zip(self.coords, other.coords, strict=False):
                out.append(coord1 - coord2)
        else:
            raise NotImplementedError(
                f"Subtraction f a CoordSet with an object of type {type(other)} is not implemented yet"
            )

        return CoordSet(list(out))

    def __add__(self, other):
        """
        Addition of Coordsets.

        Parameters
        ----------
        other : CoordSet,
            The Coordset to add to self.

        Returns
        -------
        add : CoordSet
            The sum of the Coordsets.

        """
        out = []
        if isinstance(other, CoordSet):
            for coord1, coord2 in zip(self.coords, other.coords, strict=False):
                out.append(coord1 + coord2)

        else:
            raise NotImplementedError(
                f"Addition of a CoordSet with an object of type {type(other)} is not implemented yet"
            )

        return CoordSet(list(out))

    # ----------------------------------------------------------------------------------
    # Validation methods
    # ----------------------------------------------------------------------------------
    @validate("_coords")
    def _coords_validate(self, proposal):
        coords = proposal["value"]
        if not coords:
            return []

        for id, coord in enumerate(coords):
            if coord and not isinstance(coord, Coord | CoordSet):
                raise TypeError(
                    "At this point all passed coordinates should be of type Coord or CoordSet!",
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
                    _ = coord.sizes
                except ValueError:
                    raise

                # change the internal names
                n = len(coord)
                coord._set_names(
                    [f"_{i + 1}" for i in range(n)],
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
                    "At this point all passed coordinates should have a valid name!",
                )

        if coords:
            if self._sorted:
                _sortedtuples = sorted(
                    (coord.name, coord) for coord in coords
                )  # Final sort
                coords = list(zip(*_sortedtuples, strict=False))[1]
            return list(coords)  # be sure its a list not a tuple
        return []

    @default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ----------------------------------------------------------------------------------
    # Readonly Properties
    # ----------------------------------------------------------------------------------
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

    @property
    def coords(self):
        """Coordinates in the coordset (list)."""
        return self._coords

    @property
    def has_defined_name(self):
        """True if the name has been defined (bool)."""
        return self.name != self.id

    @property
    def id(self):
        """Object identifier (Readonly property)."""
        return self._id

    @property
    def is_empty(self):
        """True if there is no coords defined (bool)."""
        return not self._coords

    @property
    def is_same_dim(self):
        """True if the coords define a single dimension (bool)."""
        return self._is_same_dim

    @property
    def references(self):
        return self._references

    @property
    def sizes(self):
        """
        Sizes of the coord object for each dimension (int or tuple of int).

        (readonly property). If the set is for a single dimension return a
        single size as all coordinates must have the same.
        """
        if not self._coords:
            return []
        _sizes = []
        for _i, item in enumerate(self._coords):
            _sizes.append(item.size)  # recurrence if item is a CoordSet

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise ValueError(
                    "Coordinates must be of the same size for a dimension with multiple "
                    "coordinates",
                )
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    # @property
    # def coords(self):  #TODO: replace with itertiems, items etc ... to simulate a dict
    #     """
    #     list - list of the Coord objects in the current coords (readonly
    #     property).
    #     """
    #     return self._coords

    @property
    def names(self):
        """Names of the coords in the current coords (list - read only property)."""
        _names = []
        if self._coords:
            for item in self._coords:
                if item.has_defined_name:
                    _names.append(item.name)
        return _names

    # ----------------------------------------------------------------------------------
    # Mutable Properties
    # ----------------------------------------------------------------------------------
    @property
    def default(self):
        """Default coordinates (Coord), or None if empty."""
        if not self._coords:
            return None
        return self._coords[self._default]

    @property
    def default_index(self):
        """Selected default coordinate index (int), or None if empty."""
        if not self._coords:
            return None
        return self._default

    @property
    def data(self):
        # in case data is called on a coordset for dimension with multiple coordinates
        # return the default coordinates data
        if not self._coords:
            return None
        return self.default.data

    @property
    def name(self):
        if self._name:
            return self._name
        return self._id

    @name.setter
    def name(self, value):
        if value is not None:
            self._name = value

    @property
    def titles(self):
        """Titles of the coords in the current coords (list)."""
        if not self._coords:
            return []
        _titles = []
        for item in self._coords:
            if isinstance(item, Coord):
                _titles.append(item.title if item.title else item.name)  # TODO:name
            elif isinstance(item, CoordSet):
                _titles.append(
                    [el.title if el.title else el.name for el in item._coords],
                )  # TODO:name
            else:
                raise ValueError("Something wrong with the titles!")
        return _titles

    @property
    def labels(self):
        """Labels of the coordinates in the current coordset (list)."""
        if not self._coords:
            return []
        return [item.labels for item in self._coords]

    @property
    def is_labeled(self):
        """Returns True if one of the coords is labeled."""
        if not self._coords:
            return False
        return any(item.is_labeled for item in self._coords)

    @property
    def units(self):
        """Units of the coords in the current coords (list)."""
        if not self._coords:
            return []
        return [item.units for item in self._coords]

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def copy(self, keepname=False):
        """
        Make a disconnected copy of the current coords.

        Returns
        -------
        object
            an exact copy of the current object

        """
        return self.__copy__()

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

    def select(self, val):
        """Select the default coord index."""
        self._default = min(max(0, int(val) - 1), len(self.names))

    def set(self, *args, **kwargs):
        """Set one or more coordinates in the current CoordSet."""
        if not args and not kwargs:
            return

        if len(args) == 1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]

        if isinstance(args, CoordSet):
            kwargs.update(args.to_dict())
            args = ()

        if args:
            self._coords = []  # reset

        for _i, item in enumerate(args[::-1]):
            item.name = self.available_names.pop()
            self._append(item)

        for k, item in kwargs.items():
            if isinstance(item, CoordSet):
                # try to keep this parameter to True!
                item._is_same_dim = True
            self[k] = item

    def set_titles(self, *args, **kwargs):
        """
        Set one or more coord title at once.

        Parameters
        ----------
        args : str(s)
            The list of titles to apply to the set of coordinates (they must be given
            according to the coordinate's name
            alphabetical order.
        **kwargs
            Keyword attribution of the titles. The keys must be valid names among the
            coordinate's name list. This
            is the recommended way to set titles as this will be less prone to errors.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's  name
        alphabetical order :
        e.g, the first title will be for the `x` coordinates, the second for the `y` ,
        etc.

        """
        if len(args) == 1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self._coords[i], CoordSet):
                self._coords[i].title = item
            elif is_sequence(item):
                for j, v in enumerate(self._coords[i]._coords):
                    v.title = item[j]

        for k, item in kwargs.items():
            self[k].title = item

    def set_units(self, *args, **kwargs):
        """
        Set one or more coord units at once.

        Parameters
        ----------
        *args : str(s)
            The list of units to apply to the set of coordinates (they must be given
            according to the coordinate's name
            alphabetical order.
        **kwargs
            Keyword attribution of the units. The keys must be valid names among the
            coordinate's name list. This
            is the recommended way to set units as this will be less prone to errors.
        force : bool, optional, default=False
            Whether or not the new units must be compatible with the current units. See
            the `Coord` .`to` method.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's name
        alphabetical order :
        e.g, the first units will be for the `x` coordinates, the second for the `y` , etc.

        """
        force = kwargs.pop("force", False)

        if len(args) == 1 and is_sequence(args[0]):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self._coords[i], CoordSet):
                self._coords[i].to(item, force=force, inplace=True)
            elif is_sequence(item):
                for j, v in enumerate(self._coords[i]._coords):
                    v.to(item[j], force=force, inplace=True)

        for k, item in kwargs.items():
            self[k].to(item, force=force, inplace=True)

    def to_dict(self):
        """
        Return a dict of the coordinates from the coordset.

        Returns
        -------
        out : dict
            A dictionary where keys are the names of the coordinates, and the values
            the coordinates themselves.

        """
        return dict(zip(self.names, self._coords, strict=False))

    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        k**warg
            Only keywords among the CoordSet.names are allowed - they denotes the name
            of a dimension.

        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self[idx] = Coord(kwargs.pop(dim), name=dim)

    # ----------------------------------------------------------------------------------
    # private methods
    # ----------------------------------------------------------------------------------
    def _slice_dims(self, dims, items):
        """
        Return a coordset sliced according to dataset dimensions and indexers.

        This lifecycle wrapper preserves the existing slicing semantics while
        keeping same-dimension multi-coordinate details inside CoordSet.
        """
        if isinstance(items, np.ndarray):
            # Fancy indexing from NDDataset can be returned as a single array.
            items = (items,)

        groups = self._lookup_groups()
        groups = self._slice_lifecycle_groups(groups, dims, items)
        return self._legacy_coordset_from_lifecycle_groups(groups)

    @staticmethod
    def _slice_lifecycle_groups(groups, dims, items):
        """Return projected groups after applying legacy dimension slicing."""
        sliced_groups = list(groups)
        coord_dims = [group.dim for group in groups if group.reference is None]
        coord_group_indexes = [
            index for index, group in enumerate(groups) if group.reference is None
        ]

        for axis, item in enumerate(items):
            name = dims[axis]
            coord_index = coord_dims.index(name)
            group_index = coord_group_indexes[coord_index]
            group = sliced_groups[group_index]
            entries = []

            for entry in group.entries:
                if entry.coord.is_empty:
                    coord = Coord(None, name=name)
                else:
                    coord = entry.coord[item]
                entries.append(replace(entry, coord=coord))

            sliced_groups[group_index] = replace(group, entries=tuple(entries))

        return tuple(sliced_groups)

    def _replace_dim(self, dim, value):
        """
        Return a coordset with one dimension coordinate replaced.

        This lifecycle wrapper preserves the existing assignment semantics used
        by ``NDDataset`` for simple coordinates and same-dimension
        multi-coordinate groups.
        """
        groups = self._lookup_groups()
        groups = self._replace_lifecycle_groups(groups, dim, value)
        return self._legacy_coordset_from_lifecycle_groups(groups)

    @staticmethod
    def _replace_lifecycle_groups(groups, dim, value):
        """Return projected groups after applying legacy dimension replacement."""
        coord_dims = [group.dim for group in groups if group.reference is None]
        coord_group_indexes = [
            index for index, group in enumerate(groups) if group.reference is None
        ]
        if dim not in coord_dims:
            raise ValueError(f"{dim!r} is not in list")
        coord_index = coord_dims.index(dim)
        group_index = coord_group_indexes[coord_index]
        replacement = CoordSet._replace_lifecycle_coord(dim, value)
        replacement_group = _coordset_to_groups(
            CoordSet(replacement, keepnames=True, sorted=False)
        )[0]

        replaced_groups = list(groups)
        replaced_groups[group_index] = replacement_group
        return tuple(replaced_groups)

    @staticmethod
    def _replace_lifecycle_coord(dim, value):
        """Normalize a replacement value using legacy replacement semantics."""
        listcoord = False
        if isinstance(value, list):
            listcoord = all(isinstance(item, Coord) for item in value)

        if listcoord:
            coord = list(CoordSet(value).to_dict().values())[0]
            coord.name = dim
            coord._is_same_dim = True
            return coord

        if isinstance(value, CoordSet):
            if len(value) > 1:
                value = CoordSet(value)
            coord = list(value.to_dict().values())[0]
            coord.name = dim
            coord._is_same_dim = True
            return coord

        if isinstance(value, Coord):
            value.name = dim
            return value

        return Coord(value, name=dim)

    def _drop_dims(self, dims, *, missing="ignore"):
        """
        Return a coordset with the given dimension coordinates removed.

        This lifecycle wrapper preserves the existing squeeze-time semantics
        while keeping missing-dimension handling inside ``CoordSet``.
        """
        if missing not in {"ignore", "raise"}:
            raise ValueError("missing must be either 'ignore' or 'raise'")

        if isinstance(dims, str):
            dims = (dims,)

        groups = self._lookup_groups()
        groups = self._drop_lifecycle_groups(groups, dims, missing=missing)
        return self._legacy_coordset_from_lifecycle_groups(groups)

    @staticmethod
    def _drop_lifecycle_groups(groups, dims, *, missing):
        """Return projected groups after applying legacy dimension dropping."""
        coord_dims = {group.dim for group in groups if group.reference is None}
        drop_dims = set()

        for dim in dims:
            if dim in coord_dims:
                drop_dims.add(dim)
                continue
            if missing == "raise":
                raise KeyError(dim)

        return tuple(
            group
            for group in groups
            if group.reference is not None or group.dim not in drop_dims
        )

    def _legacy_coordset_from_lifecycle_groups(self, groups):
        """Reconstruct the current legacy CoordSet shape from lifecycle groups."""
        return _groups_to_coordset(groups, name=self.name)

    def _reshape_dims(
        self,
        old_dims,
        old_shape,
        new_dims,
        new_shape,
        *,
        coord_policy,
        coords=None,
    ):
        """
        Return a coordset rebuilt for a reshape operation.

        This lifecycle wrapper preserves the existing reshape-time coordinate
        handling while keeping coordinate reconstruction policy inside
        ``CoordSet``.
        """
        if coord_policy == "drop":
            return None

        groups = self._lookup_groups()
        groups = self._reshape_lifecycle_groups(
            groups,
            old_dims,
            old_shape,
            new_dims,
            new_shape,
            coord_policy=coord_policy,
            coords=coords,
        )
        return _groups_to_coordset(groups)

    @staticmethod
    def _reshape_lifecycle_groups(
        groups,
        old_dims,
        old_shape,
        new_dims,
        new_shape,
        *,
        coord_policy,
        coords=None,
    ):
        """Return projected groups after applying legacy reshape policies."""
        if coords is not None:
            for dim_name, coord in coords.items():
                if dim_name not in new_dims:
                    raise ValueError(
                        f"Coordinate dim '{dim_name}' not found in new dims {new_dims}."
                    )
                if len(coord) != new_shape[new_dims.index(dim_name)]:
                    raise ValueError(
                        f"Coordinate for '{dim_name}' has length {len(coord)}, "
                        f"expected {new_shape[new_dims.index(dim_name)]}."
                    )

        coord_groups = {group.dim: group for group in groups if group.reference is None}
        new_coords = []

        if coord_policy == "strict":
            for old_idx_s, old_dim_s in enumerate(old_dims):
                old_size_s = old_shape[old_idx_s]
                matches = [i for i, size in enumerate(new_shape) if size == old_size_s]
                if len(matches) != 1:
                    raise ValueError(
                        f"strict mode: cannot unambiguously map dim "
                        f"'{old_dim_s}' (size {old_size_s}) to the new "
                        f"shape {new_shape}."
                    )
                if new_dims[matches[0]] != old_dim_s:
                    raise ValueError(
                        f"strict mode: dim '{old_dim_s}' maps to new dim "
                        f"'{new_dims[matches[0]]}' but name changed."
                    )

        for new_idx, new_dim in enumerate(new_dims):
            new_size = new_shape[new_idx]

            if coords is not None and new_dim in coords:
                new_coords.append(coords[new_dim])
                continue

            if coord_policy == "strict":
                if new_dim in coord_groups:
                    new_coords.append(
                        CoordSet._reshape_lifecycle_coord(coord_groups[new_dim])
                    )
                else:
                    new_coords.append(None)
            else:  # "safe"
                if (
                    new_dim in old_dims
                    and old_shape[old_dims.index(new_dim)] == new_size
                    and new_dim in coord_groups
                ):
                    new_coords.append(
                        CoordSet._reshape_lifecycle_coord(coord_groups[new_dim])
                    )
                else:
                    new_coords.append(None)

        return _coordset_to_groups(CoordSet(*new_coords, dims=new_dims.copy()))

    @staticmethod
    def _reshape_lifecycle_coord(group):
        """Rebuild one legacy coordinate container from one projected group."""
        coordset = _groups_to_coordset((group,))
        return coordset[group.dim].copy()

    def _reduce_dim(self, dim, *, keepdims=False):
        """
        Return a coordset updated for a standard reduction operation.

        This lifecycle wrapper preserves the existing reduction-time dimension
        cleanup while keeping coordinate handling inside ``CoordSet``.
        """
        if dim is None:
            return None

        if not keepdims:
            return self._drop_dims(dim, missing="raise")

        groups = self._lookup_groups()
        groups = self._reduce_lifecycle_groups(groups, dim)
        return self._legacy_coordset_from_lifecycle_groups(groups)

    @staticmethod
    def _reduce_lifecycle_groups(groups, dim):
        """Return projected groups after applying legacy keepdims reduction."""
        coord_dims = [group.dim for group in groups if group.reference is None]
        if dim not in coord_dims:
            raise ValueError(f"{dim!r} is not in list")

        reduced_groups = []
        for group in groups:
            if group.reference is not None or group.dim != dim:
                reduced_groups.append(group)
                continue

            reduced_entries = []
            for entry in group.entries:
                coord = entry.coord.copy(keepname=True)
                coord.data = [0]
                reduced_entries.append(replace(entry, coord=coord))

            reduced_groups.append(replace(group, entries=tuple(reduced_entries)))

        return tuple(reduced_groups)

    def _concatenate_dim(self, dim, coordsets):
        """
        Return a coordset with the concatenated dimension coordinate updated.

        This lifecycle wrapper concatenates labels and data from multiple
        coordsets for the specified dimension coordinate, preserving the
        existing concatenation-time coordinate semantics.
        """
        groups = self._lookup_groups()
        groups = self._concatenate_lifecycle_groups(groups, dim, coordsets)
        result = self._legacy_coordset_from_lifecycle_groups(groups)

        data_tuple = tuple(cs[dim].data for cs in coordsets)
        none_coord = any(x is None for x in data_tuple)
        if not none_coord:
            result[dim]._data = np.concatenate(data_tuple)
        else:
            warnings.warn(
                f"Some dataset(s) coordinates in the {dim} dimension are None.",
                stacklevel=2,
            )

        return result

    @staticmethod
    def _concatenate_lifecycle_groups(groups, dim, coordsets):
        """
        Return projected groups after concatenating dimension coordinate labels.

        This helper applies the per-entry label concatenation logic on the
        transient group projection.  Data concatenation is handled by the
        caller since it applies to the reconstructed container (Coord or
        CoordSet) rather than individual group entries.
        """
        coord_dims = [g.dim for g in groups if g.reference is None]
        if dim not in coord_dims:
            return tuple(groups)

        coord_indices = [i for i, g in enumerate(groups) if g.reference is None]
        idx = coord_dims.index(dim)
        group_idx = coord_indices[idx]
        group = groups[group_idx]

        if not group.entries or group.entries[0].coord.is_empty:
            return tuple(groups)

        new_entries = []
        for i, entry in enumerate(group.entries):
            new_coord = entry.coord.copy(keepname=True)

            per_coord_labels = []
            for cs in coordsets:
                legacy = cs[dim]
                if isinstance(legacy, CoordSet) and i < len(legacy._coords):
                    lbl = legacy._coords[i].labels
                elif i == 0:
                    lbl = legacy.labels
                else:
                    lbl = None
                if lbl is not None:
                    per_coord_labels.append(lbl)

            if per_coord_labels:
                try:
                    has_labels = np.all(
                        np.array(per_coord_labels) != [None] * len(per_coord_labels),
                    )
                except ValueError:
                    has_labels = True
                if has_labels:
                    new_coord._labels = np.concatenate(per_coord_labels)

            new_entries.append(replace(entry, coord=new_coord))

        result_groups = list(groups)
        result_groups[group_idx] = replace(group, entries=tuple(new_entries))
        return tuple(result_groups)

    def _interpolate_dim(self, dim, target_coord, *, interpolate_secondary):
        """
        Return a coordset with the interpolated dimension coordinate updated.

        This lifecycle wrapper reconstructs the coordinate container for a
        single interpolated dimension using group projection, delegating
        per-coordinate numerical interpolation to the
        ``interpolate_secondary`` callable.

        Parameters
        ----------
        dim : str
            Name of the interpolated dimension.
        target_coord : Coord
            Target coordinate for the new grid (normalized, unit-converted).
        interpolate_secondary : callable
            Callable ``(Coord) -> Coord`` that copies a secondary coordinate,
            interpolates its numeric data (when usable), clears labels, and
            returns the result.
        """
        groups = self._lookup_groups()
        groups = self._interpolate_lifecycle_groups(
            groups,
            dim,
            target_coord,
            interpolate_secondary,
        )
        return self._legacy_coordset_from_lifecycle_groups(groups)

    @staticmethod
    def _interpolate_lifecycle_groups(groups, dim, target_coord, interpolate_secondary):
        """
        Return projected groups after applying interpolation dimension updates.

        Handles three cases:
        1. **Dim not found** -- appends a new group with *target_coord*.
        2. **Simple coord** (single entry, no aliases) -- replaces the entry's
           coord with a copy of *target_coord* (labels cleared).
        3. **Same-dim multi-coord** (multiple entries or aliases) -- applies
           *interpolate_secondary* to each entry's coord.
        """
        coord_dims = [g.dim for g in groups if g.reference is None]

        if dim not in coord_dims:
            new_coord = target_coord.copy()
            new_coord._labels = None
            temp_coordset = CoordSet(new_coord, keepnames=True, sorted=False)
            new_group = _coordset_to_groups(temp_coordset)[0]
            return tuple(groups) + (new_group,)

        coord_indices = [i for i, g in enumerate(groups) if g.reference is None]
        idx = coord_dims.index(dim)
        group_idx = coord_indices[idx]
        group = groups[group_idx]

        if len(group.entries) == 1 and not group.aliases:
            new_coord = target_coord.copy()
            new_coord._labels = None
            new_entry = replace(group.entries[0], coord=new_coord)
            new_group = replace(group, entries=(new_entry,))
        else:
            new_entries = tuple(
                replace(entry, coord=interpolate_secondary(entry.coord))
                for entry in group.entries
            )
            new_group = replace(group, entries=new_entries)

        result_groups = list(groups)
        result_groups[group_idx] = new_group
        return tuple(result_groups)

    def _append(self, coord):
        # utility function to append coordinate with full validation
        if not isinstance(coord, tuple):
            coord = (coord,)
        if self._coords:
            # some coordinates already present, prepend the new one
            self._coords = (*coord,) + tuple(
                self._coords,
            )  # instead of append, fire the validation process
        else:
            # no coordinates yet, start a new tuple of coordinate
            self._coords = (*coord,)

    def _loc2index(self, loc):
        # Return the index of a location
        for coord in self.coords:
            try:
                return coord._loc2index(loc)
            except IndexError:
                continue
        # not found!
        raise IndexError

    def _set_names(self, names):
        # utility function to change names of coordinates (in batch)
        # useful when a coordinate is a CoordSet itself
        for coord, name in zip(self._coords, names, strict=False):
            coord.name = name

    def _set_parent_dim(self, name):
        # utility function to set the paretn name for sub coordset
        for coord in self._coords:
            coord._parent_dim = name

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    # @staticmethod
    def _attributes_(self):
        return ["coords", "references", "is_same_dim", "default_index", "name"]

    def __call__(self, *args, **kwargs):
        # allow the following syntax: coords(), coords(0,2) or
        coords = []
        axis = kwargs.get("axis")
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
        return CoordSet(*coords)

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    def __len__(self):
        return len(self._coords)

    def __delattr__(self, item):
        if "notify_change" in item:
            return None

        try:
            return self.__delitem__(item)
        except (IndexError, KeyError):
            raise AttributeError from None

    def __getattr__(self, item):
        # when the attribute was not found
        if "_validate" in item or "_changed" in item or item in ["strip", "__iter__"]:
            raise AttributeError

        try:
            return self.__getitem__(item)
        except (IndexError, KeyError):
            raise AttributeError from None

    # ------------------------------------------------------------------
    # Private resolver helpers (read-lookup only)
    # ------------------------------------------------------------------

    def _lookup_groups(self):
        """Project current legacy storage to private lookup groups."""
        return _coordset_to_groups(self)

    def _coordinate_lookup_groups(self):
        """Private groups that correspond positionally to legacy ``_coords``."""
        return self._filter_coordinate_lookup_groups(self._lookup_groups())

    @staticmethod
    def _filter_coordinate_lookup_groups(groups):
        """Return projected coordinate groups, excluding reference records."""
        return tuple(group for group in groups if group.reference is None)

    def _lookup_group_for_legacy_index(self, coord_groups, index):
        """Return the projected group corresponding to one legacy coord index."""
        if self.is_same_dim:
            return coord_groups[0] if coord_groups else None
        if not isinstance(index, int):
            return None
        return coord_groups[index] if index < len(coord_groups) else None

    @staticmethod
    def _lookup_entry(group, index):
        """Return one private group entry by position if available."""
        if group is None or index < 0 or index >= len(group.entries):
            return None
        return group.entries[index]

    @staticmethod
    def _lookup_entry_index(group, entry_id):
        """Return the position of one private group entry id if available."""
        if group is None or entry_id is None:
            return None
        for index, entry in enumerate(group.entries):
            if entry.id == entry_id:
                return index
        return None

    @staticmethod
    def _lookup_alias(group, entry_id):
        """Return the first compatibility alias for *entry_id* in one group."""
        if group is None:
            return None
        for alias, candidate_id in group.aliases.items():
            if candidate_id == entry_id:
                return alias
        return None

    @staticmethod
    def _lookup_reference_group(groups, dim):
        """Return the private reference group for *dim* if present."""
        for group in groups:
            if group.reference is not None and group.reference.dim == dim:
                return group
        return None

    @staticmethod
    def _lookup_coordinate_group_by_dim(coord_groups, dim):
        """Return a projected coordinate group and legacy index for *dim*."""
        for index, group in enumerate(coord_groups):
            if group.dim == dim:
                return index, group
        return None, None

    @staticmethod
    def _lookup_entry_title(entry):
        """Return the legacy title lookup key for one projected entry."""
        return entry.coord.title if entry.coord.title else entry.coord.name

    def _lookup_entry_by_alias(self, group, alias):
        """Return an entry and its position for a compatibility alias."""
        if group is None:
            return None, None
        for index, entry in enumerate(group.entries):
            if alias in entry.aliases:
                return index, entry

        entry_id = group.aliases.get(alias)
        index = self._lookup_entry_index(group, entry_id)
        return index, self._lookup_entry(group, index) if index is not None else None

    def _lookup_entry_by_title(self, group, title):
        """Return the first entry matching one legacy title lookup key."""
        if group is None:
            return None, None
        for index, entry in enumerate(group.entries):
            if self._lookup_entry_title(entry) == title:
                return index, entry
        return None, None

    @staticmethod
    def _lookup_default_entry_id(group):
        """Return the selected private entry id for one projected group."""
        if group is None:
            return None
        if group.default_id is not None:
            return group.default_id
        if len(group.entries) == 1:
            return group.entries[0].id
        return None

    def _lookup_dimension_match(self, index, coord_groups):
        """Resolve exact dimension/name lookup using projected group metadata."""
        if self.is_same_dim:
            group = coord_groups[0] if coord_groups else None
            idx, entry = self._lookup_entry_by_alias(group, index)
        else:
            idx, group = self._lookup_coordinate_group_by_dim(coord_groups, index)
            entry = None

        if idx is None:
            return None

        coord = self._coords.__getitem__(idx)
        entry_id = (
            entry.id if entry is not None else self._lookup_default_entry_id(group)
        )
        owner_dim = group.dim if group is not None else getattr(coord, "name", None)
        return _CoordLookupResult(
            coord,
            "dimension",
            owner_dim,
            index,
            entry_id=entry_id,
            alias=index if self.is_same_dim else None,
        )

    def _lookup_reference_match(self, index, groups):
        """Resolve reference lookup using projected reference metadata."""
        group = self._lookup_reference_group(groups, index)
        if group is None or group.reference is None:
            return None

        target_dim = group.reference.target_dim
        return _CoordLookupResult(
            target_dim,
            "reference",
            index,
            index,
            reference_target=target_dim,
        )

    def _lookup_top_level_title_matches(self, index, coord_groups):
        """Return top-level title matches from projected group metadata."""
        matches = []

        if self.is_same_dim:
            group = coord_groups[0] if coord_groups else None
            for idx, entry in enumerate(group.entries if group is not None else ()):
                if self._lookup_entry_title(entry) == index:
                    matches.append((idx, group, entry))
            return matches

        for idx, group in enumerate(coord_groups):
            if group.aliases or len(group.entries) != 1:
                continue
            entry = group.entries[0]
            if self._lookup_entry_title(entry) == index:
                matches.append((idx, group, entry))

        return matches

    def _lookup_top_level_title_match(self, index, coord_groups):
        """Resolve top-level title lookup while preserving warning behavior."""
        matches = self._lookup_top_level_title_matches(index, coord_groups)
        if not matches:
            return None

        warning_message = None
        if len(matches) > 1:
            warning_message = (
                f"Getting a coordinate from its title. However `{index}` occurs "
                f"several time. Only the first occurrence is returned!"
            )
            warnings.warn(warning_message, stacklevel=2)

        idx, group, entry = matches[0]
        coord = self._coords.__getitem__(idx)
        entry_id = (
            entry.id if entry is not None else self._lookup_default_entry_id(group)
        )
        owner_dim = group.dim if group is not None else getattr(coord, "name", None)
        return _CoordLookupResult(
            coord,
            "title",
            owner_dim,
            index,
            warning_message,
            entry_id=entry_id,
            alias=(
                self._lookup_alias(group, entry.id)
                if self.is_same_dim and entry is not None
                else None
            ),
        )

    def _lookup_child_title_match(self, index, coord_groups):
        """Resolve nested child title lookup using projected group entries."""
        for item, group in zip(self._coords, coord_groups, strict=False):
            if isinstance(item, CoordSet):
                idx, entry = self._lookup_entry_by_title(group, index)
            else:
                idx, entry = None, None

            if idx is not None:
                coord = item._coords.__getitem__(idx)
                return _CoordLookupResult(
                    coord,
                    "child_title",
                    group.dim,
                    index,
                    entry_id=entry.id if entry is not None else None,
                    alias=(
                        self._lookup_alias(group, entry.id)
                        if entry is not None
                        else None
                    ),
                )
        return None

    def _lookup_child_alias_match(self, index, coord_groups):
        """Resolve global nested child alias lookup using projected group entries."""
        for item, group in zip(self._coords, coord_groups, strict=False):
            if isinstance(item, CoordSet):
                idx, entry = self._lookup_entry_by_alias(group, index)
            else:
                idx, entry = None, None

            if idx is not None:
                coord = item._coords.__getitem__(idx)
                return _CoordLookupResult(
                    coord,
                    "child_name",
                    group.dim,
                    index,
                    entry_id=entry.id if entry is not None else None,
                    alias=index,
                )
        return None

    def _lookup_synthetic_alias_match(self, index, coord_groups):
        """Resolve scoped synthetic aliases while preserving legacy parsing."""
        try:
            idx, group = self._lookup_coordinate_group_by_dim(coord_groups, index[0])
            if idx is None:
                return None

            coord = self._coords.__getitem__(idx)
            owner_dim = group.dim if group is not None else getattr(coord, "name", None)
            if len(index) > 1 and index[1] == "_":
                if isinstance(coord, CoordSet):
                    result = coord._resolve_get_result(index[1:])
                    return _CoordLookupResult(
                        result.value,
                        "synthetic_alias",
                        owner_dim,
                        index,
                        result.warning_message,
                        entry_id=result.entry_id,
                        alias=result.alias,
                    )
                coord = coord.__getitem__(index[2:])  # try on labels

            return _CoordLookupResult(
                coord,
                "dimension",
                owner_dim,
                index,
                entry_id=self._lookup_default_entry_id(group),
            )
        except IndexError:
            return None

    def _resolve_string_lookup_result(self, index):
        """
        Resolve a string key using current lookup precedence.

        Precedence (highest first):
        1. top-level name
        2. reference
        3. top-level title
        4. nested child title
        5. nested child name
        6. canonical synthetic alias (e.g. ``x_1``)
        """
        groups = self._lookup_groups()
        coord_groups = self._filter_coordinate_lookup_groups(groups)

        for resolver, args in (
            (self._lookup_dimension_match, (coord_groups,)),
            (self._lookup_reference_match, (groups,)),
            (self._lookup_top_level_title_match, (coord_groups,)),
            (self._lookup_child_title_match, (coord_groups,)),
            (self._lookup_child_alias_match, (coord_groups,)),
            (self._lookup_synthetic_alias_match, (coord_groups,)),
        ):
            result = resolver(index, *args)
            if result is not None:
                return result

        raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    def _resolve_numeric_lookup_result(self, index):
        """
        Resolve a numeric key.

        For top-level CoordSets this is positional access into ``_coords``.
        For same-dimension multi-coordinate groups this slices every child.
        """
        multi = bool(self.is_same_dim)
        coord_groups = self._coordinate_lookup_groups()

        if not multi:
            coord = self._coords.__getitem__(index)
            group = self._lookup_group_for_legacy_index(coord_groups, index)
            owner_dim = group.dim if group is not None else getattr(coord, "name", None)
            return _CoordLookupResult(
                coord,
                "numeric",
                owner_dim,
                index,
                entry_id=self._lookup_default_entry_id(group),
            )

        res = []
        for c in self._coords:
            res.append(c.__getitem__(index))
        coords = self.__class__(*res, keepnames=True)
        coords.name = self.name
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        group = coord_groups[0] if coord_groups else None
        return _CoordLookupResult(
            coords,
            "numeric",
            group.dim if group is not None else self.name,
            index,
            entry_id=group.default_id if group is not None else None,
        )

    def _resolve_get_result(self, index):
        """
        Resolve *index* and return the matching private lookup result.

        Public callers should continue to use ``__getitem__`` or
        ``_resolve_get()``, which unwrap the legacy value.
        """
        if isinstance(index, str):
            return self._resolve_string_lookup_result(index)
        return self._resolve_numeric_lookup_result(index)

    def _resolve_get(self, index):
        """
        Resolve *index* and return the matching coordinate(s).

        Dispatches to ``_resolve_string_lookup`` or ``_resolve_numeric_lookup``.
        """
        return self._resolve_get_result(index).value

    # ------------------------------------------------------------------
    # Private resolver helpers (write/delete lookup)
    # ------------------------------------------------------------------

    def _sync_from_coordset(self, other):
        """Replace legacy runtime state from a reconstructed CoordSet."""
        self._coords = other._coords
        self._default = other._default
        self._is_same_dim = other._is_same_dim
        self._references = other._references

    def _sync_same_dim_from_groups(self, groups):
        """Update legacy storage for a same-dim CoordSet from updated groups."""
        group = groups[0]
        # Use in-place list mutation to avoid triggering _coords_validate,
        # which would convert empty list to None or raise on unnamed coords.
        del self._coords[:]
        for entry in group.entries:
            self._coords.append(entry.coord)
        self._default = 0
        if group.default_id:
            for i, entry in enumerate(group.entries):
                if entry.id == group.default_id:
                    self._default = i
                    break
        self._is_same_dim = True
        self._references = {}

    @staticmethod
    def _replace_group_in_groups(groups, old_group, new_group):
        """Replace one group in a groups tuple by identity."""
        return tuple(new_group if g is old_group else g for g in groups)

    @staticmethod
    def _remove_group_from_groups(groups, group):
        """Remove one group from a groups tuple by identity."""
        return tuple(g for g in groups if g is not group)

    def _resolve_set_numeric_groups(self, groups, index, coord):
        """Group-backed numeric positional assignment."""
        if isinstance(index, str):
            return None

        coord_groups = self._filter_coordinate_lookup_groups(groups)
        if not coord_groups:
            return None

        if self.is_same_dim:
            group = coord_groups[0]
            if 0 <= index < len(group.entries):
                entries = list(group.entries)
                entries[index] = replace(
                    entries[index], coord=coord.copy(keepname=True)
                )
                return self._replace_group_in_groups(
                    groups, group, replace(group, entries=tuple(entries))
                )
        else:
            if 0 <= index < len(coord_groups):
                group = coord_groups[index]
                new_entry = replace(group.entries[0], coord=coord.copy(keepname=True))
                return self._replace_group_in_groups(
                    groups,
                    group,
                    replace(group, entries=(new_entry,), aliases={}),
                )

        return None

    def _resolve_set_name_groups(self, groups, coord_groups, index, coord):
        """Group-backed top-level name replacement."""
        if self.is_same_dim:
            group = coord_groups[0] if coord_groups else None
            entry_idx, entry = self._lookup_entry_by_alias(group, index)
            if entry is not None:
                coord.name = index
                entries = list(group.entries)
                entries[entry_idx] = replace(entry, coord=coord)
                return self._replace_group_in_groups(
                    groups, group, replace(group, entries=tuple(entries))
                )
            return None

        idx, group = self._lookup_coordinate_group_by_dim(coord_groups, index)
        if idx is None:
            return None
        coord.name = index
        new_entry = replace(group.entries[0], coord=coord)
        new_group = replace(
            group,
            entries=(new_entry,),
            aliases={},
        )
        return self._replace_group_in_groups(groups, group, new_group)

    def _resolve_set_title_groups(self, groups, coord_groups, index, coord):
        """Group-backed top-level title replacement."""
        matches = self._lookup_top_level_title_matches(index, coord_groups)
        if not matches:
            return None

        if len(matches) > 1:
            warnings.warn(
                f"Getting a coordinate from its title. However `{index}` "
                f"occurs several time. Only"
                f" the first occurrence is returned!",
                stacklevel=3,
            )

        idx, group, entry = matches[0]

        if self.is_same_dim:
            coord.name = entry.aliases[0] if entry.aliases else group.dim
            entries = list(group.entries)
            entries[idx] = replace(entry, coord=coord)
            return self._replace_group_in_groups(
                groups, group, replace(group, entries=tuple(entries))
            )

        coord.name = group.dim
        new_entry = replace(group.entries[0], coord=coord)
        return self._replace_group_in_groups(
            groups,
            group,
            replace(group, entries=(new_entry,), aliases={}),
        )

    def _resolve_set_child_title_groups(self, groups, coord_groups, index, coord):
        """Group-backed nested child title replacement."""
        if self.is_same_dim:
            return None
        for group in coord_groups:
            if len(group.entries) <= 1:
                continue
            for entry_idx, entry in enumerate(group.entries):
                if self._lookup_entry_title(entry) == index:
                    coord.name = entry.aliases[0] if entry.aliases else group.dim
                    entries = list(group.entries)
                    entries[entry_idx] = replace(entry, coord=coord)
                    return self._replace_group_in_groups(
                        groups, group, replace(group, entries=tuple(entries))
                    )
        return None

    def _resolve_set_child_name_groups(self, groups, coord_groups, index, coord):
        """Group-backed nested child name (alias) replacement."""
        if self.is_same_dim:
            return None
        for group in coord_groups:
            if len(group.entries) <= 1:
                continue
            entry_idx, entry = self._lookup_entry_by_alias(group, index)
            if entry is not None:
                coord.name = index
                entries = list(group.entries)
                entries[entry_idx] = replace(entry, coord=coord)
                return self._replace_group_in_groups(
                    groups, group, replace(group, entries=tuple(entries))
                )
        return None

    def _resolve_set_synthetic_alias_groups(self, groups, coord_groups, index, coord):
        """
        Group-backed synthetic alias replacement.

        Handles both existing child replacement and append-to-group
        for synthetic patterns like ``x_3`` where ``x`` is an existing
        dimension and ``_3`` is a new synthetic child alias.
        """
        try:
            # New: same-dim raw _N append/replacement
            if self.is_same_dim and index.startswith("_"):
                alias = index
                group = coord_groups[0] if coord_groups else None
                if group is None:
                    return None
                if alias in group.aliases:
                    entry_id = group.aliases[alias]
                    for entry_idx, entry in enumerate(group.entries):
                        if entry.id == entry_id:
                            coord.name = alias
                            entries = list(group.entries)
                            entries[entry_idx] = replace(entry, coord=coord)
                            return self._replace_group_in_groups(
                                groups,
                                group,
                                replace(group, entries=tuple(entries)),
                            )
                    return None
                coord.name = alias
                new_entry = _CoordinateEntry(
                    id=_make_entry_id(coord, alias, set()),
                    coord=coord,
                    aliases=(alias,),
                )
                new_aliases = dict(group.aliases)
                new_aliases[alias] = new_entry.id
                new_entries = list(group.entries) + [new_entry]
                new_group = replace(
                    group,
                    entries=tuple(new_entries),
                    aliases=new_aliases,
                )
                return self._replace_group_in_groups(groups, group, new_group)

            # Existing: x_N pattern for non-same-dim
            idx, group = self._lookup_coordinate_group_by_dim(coord_groups, index[0])
            if idx is None:
                return None

            if len(index) > 1 and index[1] == "_":
                alias = index[1:]
                entry_id = group.aliases.get(alias)

                if entry_id is not None:
                    for entry_idx, entry in enumerate(group.entries):
                        if entry.id == entry_id:
                            coord.name = alias
                            entries = list(group.entries)
                            entries[entry_idx] = replace(entry, coord=coord)
                            return self._replace_group_in_groups(
                                groups,
                                group,
                                replace(group, entries=tuple(entries)),
                            )

                new_entry = _CoordinateEntry(
                    id=_make_entry_id(coord, alias, set()),
                    coord=coord,
                    aliases=(alias,),
                )
                new_aliases = dict(group.aliases)
                new_aliases[alias] = new_entry.id
                new_entries = list(group.entries) + [new_entry]
                new_group = replace(
                    group,
                    entries=tuple(new_entries),
                    aliases=new_aliases,
                )
                return self._replace_group_in_groups(groups, group, new_group)

            return None
        except (KeyError, IndexError):
            return None

    def _resolve_set_append_groups(self, groups, coord_groups, index, coord):
        """Group-backed append for new dimension names."""
        if isinstance(index, str) and index in self.available_names:
            coord.name = index
            new_entry = _CoordinateEntry(
                id=_make_entry_id(coord, index, set()),
                coord=coord,
            )
            new_group = _DimensionCoordinates(
                dim=index,
                entries=(new_entry,),
                default_id=new_entry.id,
            )
            return groups + (new_group,)
        return None

    def _resolve_set_groups(self, groups, index, coord):
        """
        Orchestrate group-backed assignment resolution.

        Returns transformed groups tuple if a branch matched, or None to
        fall through to the legacy append path.
        """
        if not isinstance(index, str):
            return self._resolve_set_numeric_groups(groups, index, coord)

        coord_groups = self._filter_coordinate_lookup_groups(groups)

        for resolver in (
            self._resolve_set_name_groups,
            self._resolve_set_title_groups,
            self._resolve_set_child_title_groups,
            self._resolve_set_child_name_groups,
            self._resolve_set_synthetic_alias_groups,
            self._resolve_set_append_groups,
        ):
            result = resolver(groups, coord_groups, index, coord)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Group-backed deletion resolvers (PR20)
    # ------------------------------------------------------------------

    def _resolve_delete_name_groups(self, groups, coord_groups, index):
        """Group-backed name-based deletion."""
        if self.is_same_dim:
            group = coord_groups[0] if coord_groups else None
            entry_idx, entry = self._lookup_entry_by_alias(group, index)
            if entry is not None:
                entries = list(group.entries)
                del entries[entry_idx]
                new_aliases = dict(group.aliases)
                new_aliases.pop(index, None)
                new_default_id = group.default_id
                if group.default_id == entry.id:
                    new_default_id = entries[0].id if entries else None
                new_group = replace(
                    group,
                    entries=tuple(entries),
                    aliases=new_aliases,
                    default_id=new_default_id,
                )
                return self._replace_group_in_groups(groups, group, new_group)
            return None

        idx, group = self._lookup_coordinate_group_by_dim(coord_groups, index)
        if idx is None:
            return None

        return self._remove_group_from_groups(groups, group)

    def _resolve_delete_title_groups(self, groups, coord_groups, index):
        """Group-backed title-based deletion."""
        if self.is_same_dim:
            matches = self._lookup_top_level_title_matches(index, coord_groups)
            if not matches:
                return None

            if len(matches) > 1:
                warnings.warn(
                    f"Getting a coordinate from its title. However `{index}` "
                    f"occurs several time. Only"
                    f" the first occurrence is returned!",
                    stacklevel=3,
                )

            entry_idx, group, entry = matches[0]
            entries = list(group.entries)
            del entries[entry_idx]
            alias = next(
                (a for a, eid in group.aliases.items() if eid == entry.id), None
            )
            new_aliases = dict(group.aliases)
            if alias:
                new_aliases.pop(alias, None)
            new_default_id = group.default_id
            if group.default_id == entry.id:
                new_default_id = entries[0].id if entries else None
            new_group = replace(
                group,
                entries=tuple(entries),
                aliases=new_aliases,
                default_id=new_default_id,
            )
            return self._replace_group_in_groups(groups, group, new_group)

        matches = self._lookup_top_level_title_matches(index, coord_groups)
        if not matches:
            return None

        if len(matches) > 1:
            warnings.warn(
                f"Getting a coordinate from its title. However `{index}` "
                f"occurs several time. Only"
                f" the first occurrence is returned!",
                stacklevel=3,
            )

        _, group, _ = matches[0]
        return self._remove_group_from_groups(groups, group)

    def _resolve_delete_groups(self, groups, coord_groups, index):
        """
        Orchestrate group-backed deletion resolution.

        Returns transformed groups tuple if a branch matched, or None to
        fall through to the legacy path.
        """
        for resolver in (
            self._resolve_delete_name_groups,
            self._resolve_delete_title_groups,
        ):
            result = resolver(groups, coord_groups, index)
            if result is not None:
                return result
        return None

    def _resolve_set(self, index, coord):
        """
        Resolve *index* and apply the coordinate assignment.

        Precedence (highest first):
        0. canonical synthetic alias (pre-group, preserves legacy in-place mutation)
        1. top-level name replacement
        2. top-level title replacement
        3. nested child title replacement
        4. nested child name replacement
        5. canonical synthetic alias replacement
        6. append new coordinate
        """
        try:
            coord = coord.copy(keepname=True)  # to avoid modifying the original
        except TypeError as e:
            if isinstance(coord, list):
                coord = [c.copy(keepname=True) for c in coord[:]]
            else:
                raise e

        # Canonical synthetic alias: handled before group-backed path to
        # preserve legacy in-place mutation on the child CoordSet.
        # The group reconstruction would validate same-dim sizes, but the
        # legacy code allows size changes via in-place mutation.
        if (
            isinstance(index, str)
            and len(index) > 1
            and index[1] == "_"
            and index[0] in self.names
            and isinstance(
                c := self._coords.__getitem__(self.names.index(index[0])), CoordSet
            )
        ):
            c.__setitem__(index[1:], coord)
            return

        # Group-backed resolution (preferred path)
        groups = self._lookup_groups()
        groups = self._resolve_set_groups(groups, index, coord)
        if groups is not None:
            if self.is_same_dim:
                # Same-dim CoordSets bypass _groups_to_coordset to avoid
                # double-wrapping (the reconstruction wraps the inner
                # same-dim group in an extra CoordSet layer).
                self._sync_same_dim_from_groups(groups)
                return
            try:
                result = self._legacy_coordset_from_lifecycle_groups(groups)
            except ValueError:
                # Validation failed during reconstruction (e.g., same-dim
                # size mismatch). Fall through to legacy in-place mutation.
                pass
            else:
                self._sync_from_coordset(result)
                return

        # Legacy paths (fallthrough)

        # 1. numeric index replacement
        if not isinstance(index, str):
            self._coords[index] = coord
            return

        # 1. top-level name replacement
        if index in self.names:
            idx = self.names.index(index)
            coord.name = index
            self._coords.__setitem__(idx, coord)
            return

        # 2. top-level title replacement
        if index in self.titles:
            idx = self.titles.index(index)
            coord.name = index
            self._coords.__setitem__(idx, coord)
            return

        # 3. nested child title replacement
        for item in self._coords:
            if isinstance(item, CoordSet) and index in item.titles:
                item.__setitem__(index, coord)
                return

        # 4. nested child name replacement
        for item in self._coords:
            if isinstance(item, CoordSet) and index in item.names:
                item.__setitem__(index, coord)
                return

        # 5. canonical synthetic alias (append case only; replacement is
        #    handled above before the group-backed path)
        if index[0] in self.names:
            c = self._coords.__getitem__(self.names.index(index[0]))
            if len(index) > 1 and index[1] == "_" and isinstance(c, CoordSet):
                c.__setitem__(index[1:], coord)
                return

        # 6. append new coordinate
        if isinstance(index, str) and (
            index in self.available_names
            or (
                len(index) == 2
                and index.startswith("_")
                and index[1] in list("123456789")
            )
        ):
            coord.name = index
            self._coords.append(coord)
            return

        raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    def _resolve_delete(self, index):
        """
        Resolve *index* and apply the coordinate deletion.

        Precedence (highest first):
        0. canonical synthetic alias deletion (pre-group delegation)
        1. top-level name deletion
        2. top-level title deletion
        3. nested child title deletion
        4. canonical synthetic alias deletion
        """
        if not isinstance(index, str):
            return

        # 0. Canonical synthetic alias: handled before group-backed path to
        #    preserve legacy in-place mutation on the child CoordSet.
        if (
            len(index) > 1
            and index[1] == "_"
            and index[0] in self.names
            and isinstance(
                c := self._coords.__getitem__(self.names.index(index[0])), CoordSet
            )
        ):
            c.__delitem__(index[1:])
            return

        # Group-backed resolution (preferred path)
        groups = self._lookup_groups()
        coord_groups = self._filter_coordinate_lookup_groups(groups)
        groups = self._resolve_delete_groups(groups, coord_groups, index)
        if groups is not None:
            if self.is_same_dim:
                # Same-dim CoordSets bypass _groups_to_coordset to avoid
                # double-wrapping (the reconstruction wraps the inner
                # same-dim group in an extra CoordSet layer).
                self._sync_same_dim_from_groups(groups)
                return
            if not self._filter_coordinate_lookup_groups(groups):
                # All dimension groups removed.  Sync empty state directly
                # since _groups_to_coordset does not handle empty groups.
                # Use in-place mutation (del [:] ) to avoid triggering the
                # _coords_validate validator, which would convert empty list
                # to None and break __len__().
                del self._coords[:]
                self._default = 0
                self._is_same_dim = False
                self._references = {}
                return
            try:
                result = self._legacy_coordset_from_lifecycle_groups(groups)
            except ValueError:
                pass
            else:
                self._sync_from_coordset(result)
                return

        # Legacy paths (fallthrough)

        # 1. top-level name deletion
        if index in self.names:
            idx = self.names.index(index)
            del self._coords[idx]
            return

        # 2. top-level title deletion
        if index in self.titles:
            idx = self.titles.index(index)
            self._coords.__delitem__(idx)
            return

        # 3. nested child title deletion
        for item in self._coords:
            if isinstance(item, CoordSet) and index in item.titles:
                item.__delitem__(index)
                return

        # 4. canonical synthetic alias deletion
        if index[0] in self.names:
            c = self._coords.__getitem__(self.names.index(index[0]))
            if len(index) > 1 and index[1] == "_" and isinstance(c, CoordSet):
                c.__delitem__(index[1:])
                return

        raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    def __getitem__(self, index):
        return self._resolve_get(index)

    def __setitem__(self, index, coord):
        self._resolve_set(index, coord)

    def __delitem__(self, index):
        self._resolve_delete(index)

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

    # def __iter__(self):
    #    for item in self._coords:
    #        yield item

    def __repr__(self):
        out = "CoordSet: [" + ", ".join(["{}"] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace("CoordSet: ", ""))
            else:
                s.append(f"{item.name}:{item.title}")
        return out.format(*s)

    def __str__(self):
        return repr(self)

    def _cstr(self, header="  coordinates: ... \n", print_size=True):
        txt = ""
        for _idx, dim in enumerate(self.names):
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
                            txt += f"{coord._coords[0]._str_shape().rstrip()}\n"

                        coord._html_output = self._html_output
                        for _idx_s, dim_s in enumerate(coord.names):
                            c = getattr(coord, dim_s)
                            txt += f"          ({dim_s}) ...\n"
                            c._html_output = self._html_output
                            sub = c._cstr(
                                header="  coordinates: ... \n",
                                print_size=False,
                            )  # , indent=4, first_indent=-6)
                            txt += f"{sub}\n"

                elif not coord.is_empty:
                    # coordinates if available
                    # txt += '        index: {}\n'.format(idx)
                    coord._html_output = self._html_output
                    txt += f"{coord._cstr(header=header, print_size=print_size)}\n"

        txt = txt.rstrip()  # remove the trailing '\n'

        if not self._html_output:
            return colored_output(txt.rstrip())
        return txt.rstrip()

    def _repr_html_(self):
        return convert_to_html(self)

    def __deepcopy__(self, memo):
        coords = self.__class__(
            tuple(cpy.deepcopy(ax, memo=memo) for ax in self._coords),
            keepnames=True,
        )
        coords.name = self.name
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        coords._references = cpy.deepcopy(self._references, memo=memo)
        return coords

    def __copy__(self):
        coords = self.__class__(
            tuple(cpy.copy(ax) for ax in self._coords),
            keepnames=True,
        )
        # name must be changed
        coords.name = self.name
        # and is_same_dim and default for coordset
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        coords._references = cpy.copy(self._references)
        return coords

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self._coords == other._coords
        except Exception:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # ----------------------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------------------
    def _coords_update(self, change):
        # when notified that a coord name have been updated
        self._updated = True

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
