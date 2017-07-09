# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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


#

"""This module mainly contains the definition of a Meta class object

Such object is particularly used in `SpectrochemPy` by the `NDDataset` object
to store metadata. Like a regular dictionary, the
elements can be accessed by key, but also by attributes, *e.g.*
`a = meta['key']` give the same results as `a = meta.key`.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from traitlets import HasTraits, Dict, Bool, default

import logging
log = logging.getLogger()


__all__ = ['Meta']

# =============================================================================
# Class Meta
# =============================================================================

class Meta(object): #HasTraits):
    """A dictionary to store metadata.

    The metadata are accessible by item or by attributes, and
    the dictionary can be made read-only if necessary.

    Examples
    --------

    First we initialise a metadata object

    >>> m = Meta()

    then, metadata can be set by attribute (or by key like in a regular
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
    KeyError: 'the metadata `chaine` is read only'
    >>> print(m.chaine)
    a string

    .. rubric:: Methods

    """

    # -------------------------------------------------------------------------
    # private attributes
    # -------------------------------------------------------------------------

    _data = {} #Dict()

    #@default('_data')
    #def _get_data(self):
    #    return {}

    # -------------------------------------------------------------------------
    # public attributes
    # -------------------------------------------------------------------------

    readonly = False #Bool(False)

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------
    def __init__(self):
        self._data = dict()

    def __setattr__(self, key, value):
        if key not in [ 'readonly','_data','_trait_values', '_trait_notifiers',
                        '_trait_validators', '_cross_validation_lock']:
                self[key] = value
        else:
            self.__dict__[key] = value  # to avoid a recursive call
            # we can not use self._readonly = value!

    def __getattr__(self, key):
        return self[key]

    def __setitem__(self, key, value):

        if key in ['readonly'] or key.startswith('_'):
            raise KeyError('`{}` can not be used as a metadata key'.format(key))

        elif not self.readonly:
            self._data.update({key: value})
        else:
            raise KeyError('the metadata `{}` is read only'.format(key))

    def __getitem__(self, key):
        return self._data.get(key, None)

    def __len__(self):
        return len(self._data)

    def __copy__(self):
        ret = self.__class__()
        ret.update(self._data)
        ret.readonly = self.readonly
        return ret

    def __deepcopy__(self, memo=None):
        return self.__copy__()

    def __eq__(self, other):
        return (self._data == other._data)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        for item in sorted(self._data.keys()):
            yield item

    def __str__(self):
        return str(self._data)

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    def to_dict(self):
        """Transform a metadata dictionary to a regular one.

        Returns
        -------
        dict
            A regular dictionary

        """

        return self._data

    def update(self, d):
        """Feed a metadata dictionary with the content of an another
        dictionary

        Parameters
        ----------
        d : dict-like object
            Any dict-like object can be used, such as `dict`, traits `Dict` or
            another `Meta` object.

        """

        if isinstance(d, Meta) or hasattr(d, '_data'):
            d = d.to_dict()
        if d:
            self._data.update(d)

    def copy(self):
        """ Return a disconnected copy of self.

        Returns
        -------
        meta
            A disconnected meta object identical to the original object
        """
        return self.__copy__()

    def keys(self):
        """A list of metadata contained in the object.

        Returns
        -------
        list
            A sorted key's list

        Examples
        --------
        >>> m = Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> print(m.keys())
        ['si', 'td']

        Notes
        -----
        Alternatively, it is possible to iter directly on the Meta object

        >>> m = Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> for key in m:
        ...     print(key)
        si
        td

        """
        return [key for key in self]

    def items(self):
        """A list of metadata items contained in the object.

        Returns
        -------
        list
            An item list sorted by key

        Examples
        --------
        >>> m = Meta()
        >>> m.td = 10
        >>> m.si = 20
        >>> print(m.items())
        [('si', 20), ('td', 10)]


        """
        return [(key, self[key]) for key in self]
