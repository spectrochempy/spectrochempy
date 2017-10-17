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

from traitlets import TraitType, TraitError, List, class_of

import numpy as np

_classes = ['Array', 'Range']


# =============================================================================
# Range
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

    sampling : Int [ default 1 ]
        The interval for sampling the values

    """
    klass = list
    _cast_types = (tuple,)

    # Describe the trait type
    info_text = 'an ordered interval trait'
    allow_none = True

    def __init__(self, default_value=None, sampling=1, **kwargs):

        self._sampling = sampling
        super(Range, self).__init__(trait=None, default_value=default_value,
                                    **kwargs)

    def length_error(self, obj, value):
        e = "The '%s' trait of %s instance must be of length 2 exactly," \
            " but a value of %s was specified." \
            % (self.name, class_of(obj), value)
        raise TraitError(e)

    def validate_elements(self, obj, value):
        if value is None or len(value) == 0:
            return
        length = len(value)
        if length < 2:
            self.length_error(obj, value)
        value.sort()
        return super(Range, self).validate_elements(obj, value)

    def validate(self, obj, value):

        value = super(Range, self).validate(object, value)
        value = self.validate_elements(obj, value)

        return value


# =============================================================================
# Array
# =============================================================================

class Array(TraitType):
    """A trait Array representing a np.ndarray or nd.array-like


    """

    default_value = np.array([], dtype=object)
    allow_none = True
    info_text = 'an array'

    def validate(self, obj, value):
        if isinstance(value, np.ndarray):
            return value
        elif hasattr(value, '_data'):
            return value
        self.error(obj, value)
