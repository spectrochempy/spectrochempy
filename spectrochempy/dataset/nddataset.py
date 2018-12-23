# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module implements the |NDDataset| class.

"""

__all__ = ['NDDataset']

__dataset_methods__ = []

# =============================================================================
# Standard python imports
# =============================================================================

import itertools
import textwrap
from datetime import datetime

# =============================================================================
# third-party imports
# =============================================================================
import numpy as np
from traitlets import (List, Unicode, Instance, Bool, All, Float, validate,
                       observe, default, )
import matplotlib.pyplot as plt


# =============================================================================
# Local imports
# =============================================================================
from ..utils import (SpectroChemPyWarning, numpyprintoptions,
                     get_user_and_node, set_operators, docstrings,
                     make_func_from, )
from ..extern.traittypes import Array
from ..core.projects.baseproject import AbstractProject
from .ndarray import NDArray
from .ndcoords import Coord, CoordSet
from .ndmath import NDMath
from .ndio import NDIO
from .ndplot import NDPlot
from ..application import log



# =============================================================================
# numpy print options
# =============================================================================

numpyprintoptions()


# =============================================================================
# NDDataset class definition
# =============================================================================

class NDDataset(
        NDIO,
        NDPlot,
        NDMath,
        NDArray,
):
    """
    The main N-dimensional dataset class used by |scpy|.

    """
    author = Unicode(get_user_and_node(),
                     desc='Name of the author of this dataset',
                     config=True)

    # private metadata in addition to those of the base NDArray class
    _modified = Instance(datetime)
    _description = Unicode
    _history = List(Unicode())

    _coordset = Instance(CoordSet, allow_none=True)

    _modeldata = Array(Float(), allow_none=True)

    _copy = Bool(False)
    _labels_allowed = Bool(False)  # no labels for NDDataset

    # dataset can be members of a project. (we use the abstractclass to
    # avoir circular import)
    _parent = Instance(AbstractProject, allow_none=True)

    # _ax is a hidden variable containing the matplotlib axis defined
    # for a NDArray object.
    # most generally it is accessed using the public read-only property ax

    _ax = Instance(plt.Axes, allow_none=True)

    _fig = Instance(plt.Figure, allow_none=True)

    # ------------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------------

    docstrings.delete_params('NDArray.parameters', 'labels')

    @docstrings.dedent
    def __init__(self,
                 data=None,
                 coordset=None,
                 coordunits=None,
                 coordtitles=None,
                 **kwargs):
        """
        Parameters
        ----------
        %(NDArray.parameters.no_labels)s

        coordset : An instance of |CoordSet|, optional
            `coords` contains the coordinates for the different
            dimensions of the `data`. if `coords` is provided, it must
            specified
            the `coord` and `labels` for all dimensions of the `data`.
            Multiple `coord`'s can be specified in an |CoordSet| instance
            for each dimension.

        Notes
        -----
        The underlying array in a |NDDataset| object can be accessed
        through the `data` attribute, which will return a conventional
        |ndarray|.

        Examples
        --------
        Usage by an end-user:

        >>> from spectrochempy import NDDataset # doctest: +ELLIPSIS
        SpectroChemPy's scp...
        >>> x = NDDataset([1,2,3])
        >>> print(x.data) # doctest : +NORMALIZE_WHITESPACE
        [       1        2        3]


        """
        super(NDDataset, self).__init__(data, **kwargs)

        self.parent = None

        self._modified = self._date
        self._description = ''
        self._history = []

        self.coordset = coordset
        self.coordtitles = coordtitles
        self.coordunits = coordunits

    #
    # Default values
    #
    @default('_coordset')
    def _coordset_default(self):
        return None  # CoordSet([None for dim in self.shape])

    @default('_copy')
    def _copy_default(self):
        return False

    @default('_modeldata')
    def _modeldata_default(self):
        return None

    # --------------------------------------------------------------------------
    # additional properties (not in the NDArray base class)
    # --------------------------------------------------------------------------
    @property
    def description(self):
        """
        str - Provides a description of the underlying data

        """
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def history(self):
        """
        List of strings - Describes the history of actions made on this array

        """
        return self._history

    @history.setter
    def history(self, value):
        self._history.append(value)

    @validate('_coordset')
    def _coordset_validate(self, proposal):
        coordset = proposal['value']
        if coordset is None:
            return
        for i, item in enumerate(coordset):
            if isinstance(item, NDArray) and \
                    not isinstance(item, Coord):
                coordset[i] = Coord(item)
        return coordset

    @property
    def coordset(self):
        """
        |CoordSet| instance - Contains the coordinates of the various
        dimensions of the dataset.

        """
        return self._coordset

    @coordset.setter
    def coordset(self, value):

        if value is not None:
            if self._coordset is not None:
                log.info("Overwriting NDDataset's current "
                         "coordset with one specified")

            if not isinstance(value, CoordSet):
                value = CoordSet(value)

            coordset = CoordSet(
                    [[None] for s in self._data.shape])  # basic coordset

            for i, item in enumerate(value[::-1]):
                coordset[self._data.ndim - 1 - i] = item

            for i, coord in enumerate(coordset):

                if isinstance(coord, CoordSet):
                    size = coord.sizes[i]
                else:
                    size = coord.size
                if self.has_complex_dims and self._is_complex[i]:
                    size = size * 2
                if size != self._data.shape[i]:
                    raise ValueError(
                            'the size of each coordinates array must '
                            'be equal to that of the respective data dimension')

            self._coordset = coordset

    @property
    def coordtitles(self):
        """
        list - list of the |Coord| titles.

        """
        if self.coordset is not None:
            return self.coordset.titles

    @coordtitles.setter
    def coordtitles(self, value):
        if self.coordset is not None:
            self.coordset.titles = value

    @property
    def coordunits(self):
        """
        list - list of the |Coord| units

        """
        if self.coordset is not None:
            return self.coordset.units

    @coordunits.setter
    def coordunits(self, value):

        if self.coordset is not None:
            self.coordset.units = value

    @property
    def modeldata(self):
        """
        |ndarray| - models data eventually generated by modelling of the data

        """
        return self._modeldata

    @modeldata.setter
    def modeldata(self, data):
        self._modeldata = data

    @property
    def T(self):
        """
        |NDDataset| - Transposed array.

        The same object is returned if `ndim` is less than 2.

        """
        return self.transpose()

    @property
    def x(self):
        """
        \Coord| instance - The x coord, i.e. coordset(-1).

        If the dataset has no axis (coordset is None), then the index of the
        data is returned except if there is labels taht can replace this index.

        """
        if self.coordset is None or self.coordset[-1].size == 0: # no axis
            new = Coord( range(self.shape[-1]), title = 'index')
            return new
        return self.coordset[-1]

    @x.setter
    def x(self, value):
        self.coordset[-1] = value

    @property
    def y(self):
        """
        \Coord| instance - The y coord, i.e. coordset(-2) for 2D dataset.

        """
        if self.ndim < 2:
            warnings.warn("The data are not 2D, there is no y dim",
                          SpectroChemPyWarning)
            return None
            # if self.coordset[-2].data.size == 0:
            #     new = self.coordset[-2].copy()
            #     new._data = range(self.shape[-2])
            #     return new
            # return self.coordset[-2]
        if self.coordset is None or self.coordset[-2].size == 0:  # no axis
            new = Coord(range(self.shape[-1]), title='index')
            return new
        return self.coordset[-2]

    @y.setter
    def y(self, value):
        self.coordset[-2] = value

    @property
    def z(self):
        """
        \Coord| instance - The z coord, i.e. coordset(-3) for 3D dataset.

        """
        if self.ndim > 2:
            if self.coordset[-3].data.size == 0:
                new = self.coordset[-3].copy()
                new._data = range(self.shape[-3])
                return new
            return self.coordset[-3]

    @z.setter
    def z(self, value):
        self.coordset[-3] = value

    @property
    def modified(self):
        """
        `Datetime` object - Date of modification (readonly property).

        """
        return self._modified

    # ........................................................................
    @property
    def parent(self):
        """
        |Project| instance - The parent project of the dataset.

        """
        return self._parent

    # ........................................................................
    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this dataset but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will first remove it
            # from the current project.
            self._parent.remove_dataset(self.name)
        self._parent = value

    # -------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # usefull for this Coord class
    # -------------------------------------------------------------------------

    @property
    def labels(self):
        # not valid for NDDataset
        raise ValueError("There is no label for nd-dataset")

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    def implements(self, name=None):
        # For compatibility with pyqtgraph
        # Rather than isinstance(obj, NDDataset) use object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['NDDataset']
        else:
            return name == 'NDDataset'

    def coord(self, axis=-1):
        """
        Returns the coordinates along the given axis

        Parameters
        ----------
        axis : int or `unicode`
            An axis index or name, default=-1 for the last axis

        Returns
        -------
        |Coord|
            Coordinates along the given axis



        """
        return self.coordset[axis]

    # .........................................................................
    @docstrings.dedent
    def plus_minus(self, uncertainty, inplace=False):
        """
        Set the uncertainty of a |NDDataset|

        Parameters
        -----------
        uncertainty: float or |ndarray|
            Uncertainty to apply to the array. If it's an array, it must have
            the same shape as the `data` shape.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s. The object will be the original
        object with uncertainty.

        Examples
        --------
        >>> np.random.seed(12345)
        >>> ndd = NDArray( data = np.random.random((3)))
        >>> ndd.plus_minus(.2)
        NDArray: [   0.930+/-0.200,    0.316+/-0.200,    0.184+/-0.200] unitless
        >>> ndd # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        NDArray: [   0.930,  ...  0.184] unitless

        >>> np.random.seed(12345)
        >>> ndd = NDArray( data = np.random.random((3,3)), units='m')
        >>> ndd.plus_minus(.2, inplace=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        NDArray: [[   0.930+/-0.200,    0.316+/-0.200,    0.184+/-0.200],
              [   0.205+/-0.200,    0.568+/-0.200,    0.596+/-0.200],
              [   0.965+/-0.200,    0.653+/-0.200,    0.749+/-0.200]] m
        >>> print(ndd) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[   0.930+/-0.200    ...  0.749+/-0.200]] m

        """
        if inplace:
            new = self
        else:
            new = self.copy()

        if isinstance(uncertainty, float):
            new.uncertainty = np.ones(new._data.shape) * uncertainty
        else:
            new.uncertainty = uncertainty

        return new

    @docstrings.dedent
    def transpose(self, axes=None, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        Parameters
        ----------
        axes : list of int, optional
            By default, reverse the dimensions, otherwise permute the coordset
            according to the values given.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        See Also
        --------
        swapaxes

        """
        if not inplace:
            new = self.copy()
        else:
            new = self

        if self.ndim < 2:  # cannot transpose 1D data
            return new

        if axes is None:
            axes = list(range(self.ndim - 1, -1, -1))

        new._data = np.transpose(new._data, axes)
        if new.is_masked:
            new._mask = np.transpose(new._mask, axes)
        if new.is_uncertain:
            new._uncertainty = np.transpose(new._uncertainty, axes)
        if new._coordset is not None:
            new._coordset._transpose(axes)
        new._is_complex = [new._is_complex[axis] for axis in axes]

        return new

    @docstrings.dedent
    def swapaxes(self, axis1, axis2, inplace=False):
        """
        Interchange two dimension of a NDDataset.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        See Also
        --------
        transpose

        """

        new = super(NDDataset, self).swapaxes(axis1, axis2, inplace)

        # in addition to what has been done in NDArray base class
        # we need to also swap coordset.

        if new._coordset:
            new._coordset[axis1], new._coordset[axis2] = \
                new._coordset[axis2], new._coordset[axis1]

        return new

    @docstrings.dedent
    def sort(self,
             axis=0, pos=None, by='value', descend=False, inplace=False):
        """
        Returns the dataset sorted along a given dimension
        (by default, the first dimension [axis=0]) using the numeric or label
        values

        Parameters
        ----------
        axis : int , optional, default = 0
            axis id along which to sort.
        pos: int , optional
            If labels are multidimensional  - allow to sort on a define
            row of labels: labels[pos]. Experimental: Not yet checked
        by : str among ['value', 'label'], optional, default = ``value``.
            Indicate if the sorting is following the order of labels or
            numeric coord values.
        descend : `bool`, optional, default = `False`.
            If true the dataset is sorted in a descending direction.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        """

        if not inplace:
            new = self.copy()
        else:
            new = self

        if axis == -1:
            axis = self.ndim - 1

        indexes = []
        for i in range(self.ndim):
            if i == axis:
                if self.coordset[axis].size == 0:
                    # sometimes we have only label for Coord objects.
                    # in this case, we sort labels if they exist!
                    if self.coordset[axis].is_labeled:
                        by = 'label'
                    else:
                        # nothing to do for sorting
                        # return self itself
                        return self

                args = self.coordset[axis]._argsort(by=by, pos=pos,
                                                    descend=descend)
                new.coordset[axis] = self.coordset[axis]._take(args)
                indexes.append(args)
            else:
                indexes.append(slice(None))

        new._data = new._data[indexes]
        if new.is_masked:
            new._mask = new._mask[indexes]
        if new.is_uncertain:
            new._uncertainty = new._uncertainty[indexes]

        return new

    def set_complex(self, axis=-1):
        """
        Make a dimension complex

        Parameters
        ----------
        axis : int, optional, default:-1
            The axis to make complex

        """
        # override the ndarray function because we must care about the axis too.

        if self._data.shape[axis] % 2 == 0:
            # we have a pair number of element along this axis.
            # It can be complex
            # data are then supposed to be interlaced (real, imag, real, imag ..
            self._is_complex[axis] = True
        else:
            raise ValueError('The odd size along axis {} is not compatible with'
                             ' complex interlaced data'.format(axis))

        if self.coordset:
            new_axis = self.coordset[axis][::2]
            self.coordset[axis] = new_axis

    # Create the returned values of functions should be same class as input.
    # The units should have been handled by __array_wrap__ already

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        return NDIO().__dir__() + ['data', 'mask', 'units', 'uncertainty',
                                   'meta', 'name', 'title', 'is_complex',
                                   'coordset', 'description', 'history', 'date',
                                   'modified', 'modeldata'
                                   ]

    def __str__(self):
        # Display the metadata of the object and partially the data

        # print field names/values (class/sizes)
        # data.name, .author, .date,
        out = ''
        out += '      name/id: {}\n'.format(self.name)
        out += '       author: {}\n'.format(self.author)
        out += '      created: {}\n'.format(self._date)
        out += 'last modified: {}\n'.format(self._modified)

        wrapper1 = textwrap.TextWrapper(initial_indent='',
                                        subsequent_indent=' ' * 15,
                                        replace_whitespace=True)

        pars = self.description.strip().splitlines()

        out += '  description: '
        if pars:
            out += '{}\n'.format(wrapper1.fill(pars[0]))
        for par in pars[1:]:
            out += '{}\n'.format(textwrap.indent(par, ' ' * 15))

        if not out.endswith('\n'):
            out += '\n'

        if self._history != ['']:
            pars = self.history
            out += '      history: '
            if pars:
                out += '{}\n'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                out += '{}\n'.format(textwrap.indent(par, ' ' * 15))

            if not out.endswith('\n'):
                out += '\n'

        # if uncertainty is not nouncertainty:
        #    uncertainty = "(+/-%s)" % self.uncertainty

        # units = '{:~K}'.format(
        #         self.units) if self.units is not None else 'unitless'

        sh = ' size' if self.ndim < 2 else 'shape'
        shapecplx = (x for x in
                     itertools.chain.from_iterable(
                             list(zip(self.shape,
                                      [False] * self.ndim
                                      if not self.is_complex else self.is_complex))))
        shape = (' x '.join(['{}{}'] * self.ndim)).format(
                *shapecplx).replace(
                'False', '').replace('True', '(complex)')
        size = self.size
        sizecplx = '' if not self.has_complex_dims else " (complex)"

        out += '   data title: {}\n'.format(self.title)
        out += '    data size: {}{}\n'.format(size, sizecplx) if self.ndim < 2 \
            else '   data shape: {}\n'.format(shape)

        # out += '   data units: {}\n'.format(units)
        # data_str = str(
        #        self._uarray(self._data, self._uncertainty)).replace('\n\n',
        #                                                 '\n')

        data_str = super(NDDataset, self).__str__()

        out += '  data values:\n'
        out += '{}\n'.format(textwrap.indent(str(data_str), ' ' * 9))

        if self.coordset is not None:
            idx = 0
            for coord in self.coordset:
                if isinstance(coord, list) or (hasattr(coord, 'data')
                                               and coord.data is not None):
                    coord_str = str(coord).replace('\n\n', '\n')
                    out += 'coordinates {}:\n'.format(idx)
                    out += textwrap.indent(coord_str, ' ' * 9)
                    out += '\n'
                    idx += 1

        if not out.endswith('\n'):
            out += '\n'
        out += '\n'

        return out

    def __getattr__(self, item):
        # when the attribute was not found

        if item in ["__numpy_ufunc__"] or '_validate' in item or \
                        '_changed' in item:
            # raise an error so that masked array will be handled correctly
            # with arithmetic operators and more
            raise AttributeError

    def __eq__(self, other, attrs=None):
        attrs = self.__dir__()
        for attr in ('name', 'description', 'history', 'date', 'modified'):
            attrs.remove(attr)
        # some attrib are not important for equality
        return super(NDDataset, self).__eq__(other, attrs)

    # .........................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return super().__hash__ + hash(self._coordset)

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    def _repr_html_(self):
        tr = "<tr style='border: 1px solid lightgray;'>" \
             "<td style='padding-right:5px; width:100px'><strong>{}</strong></td>" \
             "<td style='text-align:left'>{}</td><tr>\n"

        out = "<table style='width:100%'>\n"

        out += tr.format("Name/Id", self.name)
        out += tr.format("Author", self.author)
        out += tr.format("Created", str(self.date))
        out += tr.format("Last Modified", self.modified)
        out += tr.format("Description",
                         self.description.replace('\n', '<br/>'))

        if self.history:
            pars = self.history
            hist = ""
            if pars:
                hist += '{}'.format(pars[0])
            for par in pars[1:]:
                hist += '<br/>{}'.format(par)
            out += tr.format("History", hist)

        data = "<table style='width:100%'>\n"
        data += tr.format("Title", self.title)

        sh = 'Size' if self.ndim < 2 else 'Shape'
        shapecplx = (x for x in
                     itertools.chain.from_iterable(
                             list(zip(self.shape,
                                      [False] * self.ndim
                                      if not self.is_complex else self.is_complex))))

        shape = (' x '.join(['{}{}'] * self.ndim)).format(
                *shapecplx).replace(
                'False', '').replace('True', '(complex)')

        size = self.size
        sizecplx = '' if not self.has_complex_dims else " (complex)"
        sizetxt = '{}{}'.format(size, sizecplx) if self.ndim < 2 \
            else '{}'.format(shape)

        data += tr.format(sh, sizetxt)

        data_str = super(NDDataset, self)._repr_html_()
        data += tr.format("Values", data_str)

        data += '</table>\n'  # end of row data

        out += tr.format('data', data)

        if self.coordset is not None:
            for i, coord in enumerate(self.coordset):
                coord_str = coord._repr_html_()
                out += tr.format("Coordinate %i" % i, coord_str)
                if out.endswith("\n\n"):
                    out = out[:-1]
        out += '</table><br/>\n'

        return out

    def _loc2index(self, loc, axis):
        # Return the index of a location (label or coordinates) along the axis

        coord = self.coordset[axis]
        return coord._loc2index(loc)

    # -------------------------------------------------------------------------
    # events
    # -------------------------------------------------------------------------
    @observe(All)
    def _anytrait_changed(self, change):

        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change['name'] in ["_date", "_modified", "trait_added"]:
            return

        # changes in data -> update dates
        if change['name'] == '_data' and self._date == datetime(1970, 1, 1, 0,
                                                                0):
            self._date = datetime.now()
            self._modified = datetime.now()

        # change to complex
        # change type of data to complex
        # require modification of the coordset, if any
        if change['name'] == '_is_complex':
            pass

        # all the time -> update modified date
        self._modified = datetime.now()

        return

# =============================================================================
# module function
# =============================================================================

# make some functions also accessible from the scp API
# We want a slightly different docstring so we cannot just make:
#     func = NDDataset.func

sort = make_func_from(NDDataset.sort, first='dataset')
swapaxes = make_func_from(NDDataset.swapaxes, first='dataset')
transpose = make_func_from(NDDataset.transpose, first='dataset')

abs = make_func_from(NDDataset.abs, first='dataset')
conjugate = make_func_from(NDDataset.conjugate, first='dataset')  # defined in ndarray
set_complex = make_func_from(NDArray.set_complex, first='dataset')

__all__ += ['sort',
            'swapaxes',
            'transpose',
            'abs',
            'conjugate',
            'set_complex',
            ]

# =============================================================================
# Set the operators
# =============================================================================

set_operators(NDDataset, priority=50)

if __name__ == '__main__':
    pass



