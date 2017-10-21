# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

from traitlets import TraitType, TraitError, List, class_of, HasTraits
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils import is_sequence

import numpy as np
import copy

_classes = ['Array', 'Range']


# =============================================================================
# Range trait type
# =============================================================================

class Range(List):
    """
    Create a trait with two values defining an ordered range of values,
    with an optional sampling parameters

    Parameters
    ----------

    trait : TraitType [ optional ]
        the type for restricting the contents of the Container.
        If unspecified, types are not checked.

    default_value : SequenceType [ optional ]
        The default value for the Trait.  Must be list/tuple/set, and
        will be cast to the container type.


    Examples
    --------

    >>> class MyClass(HasTraits):
    ...     r = Range([10,5])  # Initialized with some default values

    >>> c = MyClass()
    >>> print(c.r) # the list is ordered
    [5, 10]
    >>> c.r = [1, 3, 5]
    Traceback (most recent call last):
     ...
    traitlets.traitlets.TraitError: The 'r' trait of a type instance must be of length 2 exactly, but a value of [1, 3, 5] was specified.

    """
    klass = list
    _cast_types = (tuple,)

    # Describe the trait type
    info_text = 'an ordered interval trait'
    allow_none = True

    def __init__(self, default_value=None, **kwargs):

        super(Range, self).__init__(trait=None, default_value=default_value,
                                    **kwargs)
        pass

    def length_error(self, obj, value):
        e = "The '%s' trait of %s instance must be of length 2 exactly," \
            " but a value of %s was specified." \
            % (self.name, class_of(obj), value)
        raise TraitError(e)

    def validate_elements(self, obj, value):
        if value is None or len(value) == 0:
            return
        length = len(value)
        if length !=2:
            self.length_error(obj, value)
        value.sort()
        value = super(Range, self).validate_elements(obj, value)
        return value

    def validate(self, obj, value):

        value = super(Range, self).validate(object, value)
        value = self.validate_elements(obj, value)

        return value

    def __str__(self):
        return 'range %s'%str(s)



# =============================================================================
# hcarray object
# =============================================================================
# subclass of ndarray type
class hcarray(np.ndarray):
    """
    This is a subclass of the ndarray with attributes
    allowing the handling of hypercomplex data

    Examples
    --------

    Initialize with real data
    >>> x = np.array(range(4), dtype=float)
    >>> y = hcarray(x)
    >>> y  # repr()
    hcarray([ 0.,  1.,  2.,  3.])
    >>> print(y) # str()
    [ 0.  1.  2.  3.]

    >>> y.is_complex
    [False]

    >>> y.make_complex(-1)
    >>> y.is_complex
    [True]

    >>> x = np.array([[0, 1.+1.5j],[1-2.j, 2.+1j]], dtype=np.complex128)
    >>> y = hcarray(x)
    >>> y
    hcarray([[ 0. ,  0. ,  1. ,  1.5],
             [ 1. , -2. ,  2. ,  1. ]])
    >>> print(y)
    [[ 0.+0.j   1.+1.5j]
     [ 1.-2.j   2.+1.j ]]

    >>> y.is_complex
    [False, True]

    >>> x = np.array([[0, 1.+1.5j],[1-2.j, 2.+1j]], dtype=np.complex128)
    >>> y = hcarray(x, is_complex=[True, True])
    >>> y.is_complex
    [True, True]
    >>> print(y)
    [[ 0.+0.j   1.+1.5j]
     [ 1.-2.j   2.+1.j ]]
    >>> print(y.real)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RR)
    [ 0.  1.]
    >>> print(y.RI)
    [ 0.   1.5]
    >>> print(y.IR)
    [ 1.  2.]
    >>> print(y.II)
    [-2.  1.]

    Now make dimension 0 real.  Note that the dimension are not change,
    we just force all item to be real components
    >>> y.make_real(0)
    >>> y.is_complex
    [False, True]
    >>> print(y.real)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RR)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RI)
    [[ 0.   1.5]
     [-2.   1. ]]

    >>> # np.sqrt(y) This can work with this hcarray #TODO: work with the ufunc



    """
    def __new__(cls, input_array, is_complex=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        cls.x = x = np.asarray(input_array).view(cls)

        if is_complex is None:
            is_complex = [False] * x.ndim  # by default real in all dimnension

        if is_sequence(is_complex) and len(is_complex)!=x.ndim:
            raise ValueError("size of the `is_complex` list argument "
                             "doesn't match the nb of dims")

        if is_sequence(is_complex) and (x.dtype == complex):
            is_complex[-1] = True

        if x.dtype != complex:
            x = x.astype(np.float64)  # be sure all element are float or complex

        # reshape so there is no more complex data... They will be interleaved
        if x.ndim>1:
            obj = x.view(dtype=np.float64).reshape(x.shape[0],-1)
        else:
            obj = x.view(dtype=np.float64).reshape(-1)

        for i in range(x.ndim):
            if is_complex[i] and x.shape[i] % 2 != 0:
                raise ValueError("Size of complex indirect dimension "
                                 "must be even, not %d"%x.shape[i])

        # add the new attribute to the created instance
        obj.is_complex = is_complex

        # Finally, we must return the newly created object:
        return obj

    def make_complex(self, axis):
        if self.shape[axis] % 2 != 0:
            raise ValueError("Size of complex indirect dimension "
                             "must be even, not %d"%self.shape[axis])
        self.is_complex[axis] = True

    def make_real(self, axis):
        self.is_complex[axis] = False
        pass

    @property
    def real(self):
        r = self[..., ::2]
        r.is_complex[-1]=False
        return r.squeeze()

    @property
    def imag(self):
        r = self[..., 1::2]
        r.is_complex[-1]=False
        return r.squeeze()

    def part(self, select='ALL'):
        ar = self.copy()
        if select=='ALL':
            select = 'R'*self.ndim
        for axis, component in enumerate(select):
            if self.is_complex[axis]:
                ar = ar.swapaxes(axis,-1)
                if component == 'R':
                    ar = ar[..., ::2]
                elif component == 'I':
                    ar = ar[..., 1::2]
                ar = ar.swapaxes(axis, -1)
                ar.is_complex[axis] = False
        return ar.squeeze()

    @property
    def RR(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('RR')

    @property
    def RI(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('RI')

    @property
    def IR(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('IR')

    @property
    def II(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('II')

    @property
    def trueshape(self):
        return self.part('ALL').shape

    @property
    def truesize(self):
        return self.part('ALL').size

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.is_complex = copy.copy(getattr(obj, 'is_complex', None))

    def __str__(self):
        if self.is_complex[-1]:
            ar = np.array(self.real + self.imag*1j)
        else:
            ar = np.array(self)
        return ar.__str__()

# =============================================================================
# HyperComplexArray trait type
# =============================================================================

class HyperComplexArray(Array):
    """A HyperComplexArray trait representing a np.ndarray or nd.array-like

    Examples
    --------

    >>> class MyClass(HasTraits):
    ...
    ...     r = HyperComplexArray(allow_none=True)

    Initialize with a ndarray (default)
    >>> mc = MyClass(r=[1])
    >>> print(mc.r)
    [ 1.]
    >>> mc.r.is_complex
    [False]

    Initialize with the list using casting
    >>> mc = MyClass(r=[1,2])
    >>> print(mc.r)
    [ 1.  2.]


    Until there, there is no difference with Array traits
    Let's try with complex data
    Initialize with the list using casting
    >>> mc = MyClass(r=[1+0j,2+1j])
    >>> mc.r # internal representation
    hcarray([ 1.,  0.,  2.,  1.])
    >>> print(mc.r)
    [ 1.+0.j  2.+1.j]
    >>> mc.r.is_complex
    [True]

    Cast using another HyperComplexArray
    >>> r3 = mc.r.copy() * 2
    >>> mc.r = r3
    >>> mc.r
    hcarray([ 2.,  0.,  4.,  2.])
    >>> mc.r.is_complex
    [True]

    """

    klass = hcarray
    _cast_types = (np.ndarray, list, tuple,)

    # Describe the trait type
    info_text = 'an hypercomplex array trait'
    allow_none = True

    def set(self, obj, value):
        is_complex = None
        if isinstance(value, hcarray):
            is_complex = value.is_complex
        new_value = hcarray(self._validate(obj, value), is_complex)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if not np.array_equal(old_value, new_value):
            obj._notify_trait(self.name, old_value, new_value)
