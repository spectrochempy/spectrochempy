# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements the class |Coord|.
"""

__all__ = ['Coord']

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------
import textwrap

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from traitlets import Bool, observe, All, Unicode

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------
from .ndarray import NDArray
from .ndmath import NDMath, set_operators
from ...core import info_, debug_, error_, warning_
from ...utils import (docstrings, colored_output, NOMASK, spacing)


# ======================================================================================================================
# Coord
# ======================================================================================================================
class Coord(NDMath, NDArray):
    """Coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord| object.
    This is a single dimension array with either numerical (float) values or
    labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    """
    _copy = Bool

    _html_output = False
    _parent_dim = Unicode(allow_none=True)

# ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    docstrings.delete_params('NDArray.parameters', 'data', 'mask')

    # ..................................................................................................................
    @docstrings.dedent
    def __init__(self, data=None, **kwargs):
        """
        Parameters
        -----------
        data : ndarray, tuple or list
            The actual data array contained in the |Coord| object.
            The given array (with a single dimension) can be a list,
            a tuple, a |ndarray|, or a |ndarray|-like object.
            If an object is passed that contains labels, or units,
            these elements will be used to accordingly set those of the
            created object.
            If possible, the provided data will not be copied for `data` input,
            but will be passed by reference, so you should make a copy the
            `data` before passing it in the object constructor if that's the
            desired behavior or set the `copy` argument to True.
        %(NDArray.parameters.no_data|mask)s

        Examples
        --------
        We first import the object from the api :
        
        >>> from spectrochempy import *
        
        We then create a numpy |ndarray| and use it as the numerical `data`
        axis of our new |Coord| object.
        
        >>> arr = np.arange(1.,12.,2.)
        >>> c0 = Coord(data=arr, title='frequency', units='Hz')
        >>> c0     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Coord: [   1.000,    3.000,    5.000,    7.000,    9.000,   11.000] Hz
        
        We can take a series of str to create a non numerical but labelled
        axis :
        
        >>> tarr = list('abcdef')
        >>> tarr
        ['a', 'b', 'c', 'd', 'e', 'f']
        
        >>> c1 = Coord(labels=tarr, title='mylabels')
        >>> c1   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Coord: [a, b, c, d, e, f]
        
        >>> print(c1) # doctest: +NORMALIZE_WHITESPACE
        title : Mylabels
        labels : [a b c d e f]
        
        """
        super(Coord, self).__init__(data, **kwargs)

        assert self.ndim <= 1

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `Coord`.
        
        Rather than isinstance(obj, Coord) use object.implements('Coord').
        
        This is useful to check type without importing the module
        
        """
    
        if name is None:
            return 'Coord'
        else:
            return name == 'Coord'

    # ------------------------------------------------------------------------------------------------------------------
    # readonly property
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @property
    def reversed(self):
        """bool - Whether the axis is reversed (readonly
        property).
        """
        if self.units in ['1 / centimeter','ppm']:
            return True
        return False
        ## Return a correct result only if the data are sorted
        # return bool(self.data[0] > self.data[-1])

    # ------------------------------------------------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some of the property of NDArray has to be hidden because they
    # are not useful for this Coord class
    # ------------------------------------------------------------------------------------------------------------------

    # NDarray methods

    # ..................................................................................................................
    @property
    def is_complex(self):
        return False  # always real

    # ..................................................................................................................
    @property
    def ndim(self):
        ndim = super().ndim
        if ndim > 1:
            raise ValueError("Coordinate's array should be 1-dimensional!")
        return 1

    # ..................................................................................................................
    @property
    def T(self):  # no transpose
        return self

    # ..................................................................................................................
    @property
    def date(self):
        return None

    # ..................................................................................................................
    # @property
    # def values(self):
    #    return super().values

    # ..................................................................................................................
    @property
    def masked_data(self):
        return super().masked_data

    # ..................................................................................................................
    @property
    def is_masked(self):
        return False

    # ..................................................................................................................
    @property
    def mask(self):
        return super().mask

    # ..................................................................................................................
    @mask.setter
    def mask(self, val):
        # Coordinates cannot be masked. Set mask always to NOMASK
        self._mask = NOMASK

    # ..................................................................................................................
    @property
    def spacing(self):
        # return a scalar for the spacing of the coordinates (if they are uniformly spaced,
        # else return an array of the differents spacings
        return spacing(self.data) * self.units
    
    
    # NDmath methods

    # ..................................................................................................................
    def cumsum(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def mean(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def pipe(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def remove_masks(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def std(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def sum(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def swapaxes(self, **kwargs):
        raise NotImplementedError

    
    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------
    def loc2index(self, loc):
        return self._loc2index(loc)
    
    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __copy__(self):
        res = self.copy(deep=False)  # we keep name of the coordinate by default
        res.name = self.name
        return res

    # ..................................................................................................................
    def __deepcopy__(self, memo=None):
        res = self.copy(deep=True, memo=memo)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        # dtype must stay first item
        return ['data', 'labels', 'units', 'meta', 'title', 'name', 'origin']

    # ..................................................................................................................
    def __getitem__(self, items, return_index=False):
        # we need to keep the names when copying coordinates to avoid later problems
        res = super().__getitem__(items, return_index=return_index)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    # ..................................................................................................................
    def _cstr(self, header='  coordinates: ... \n', print_size=True, **kwargs):
    
        indent = kwargs.get('indent',0)
        
        out = ''
        if not self.is_empty and print_size:
            out += f'{self._str_shape().rstrip()}\n'
        out += f'        title: {self.title}\n' if self.title else ''
        if self.has_data:
            out += '{}\n'.format(self._str_value(header=header))
        elif self.is_empty and not self.is_labeled:
            out += header.replace('...', '\0Undefined\0')

        if self.is_labeled:
            header = '       labels: ... \n'
            text = str(self.labels.T).strip()
            if '\n' not in text:  # single line!
                out += header.replace('...', '\0\0{}\0\0'.format(text))
            else:
                out += header
                out += '\0\0{}\0\0'.format(textwrap.indent(text.strip(), ' ' * 9))

        if out[-1] == '\n':
            out = out[:-1]

        if indent:
            out = "{}".format(textwrap.indent(out, ' ' * indent))
        
        first_indent=kwargs.get("first_indent",0)
        if first_indent < indent:
            out = out[indent-first_indent:]
            
        if not self._html_output:
            return colored_output(out)
        else:
            return out

    # ..................................................................................................................
    def __repr__(self):
        out = self._repr_value().rstrip()
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @observe(All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }
        #debug_(f'changes in Coord: {change.name}')
        pass


# ======================================================================================================================
# Set the operators
# ======================================================================================================================
set_operators(Coord, priority=50)


# ======================================================================================================================
if __name__ == '__main__':
    pass
