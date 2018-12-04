# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



from traitlets import (TraitType, TraitError, List, class_of, HasTraits)

__all__ = []

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

# =============================================================================
# ArrayList trait type
# =============================================================================

class ArrayList(List):
    """
    Create a trait with an ordered list of same types

    Parameters
    ----------
    trait : TraitType [ optional ]
        the type for restricting the contents of the Container.
        If unspecified, types are not checked.

    default_value : SequenceType [ optional ]
        The default value for the Trait.  Must be list/tuple/set, and
        will be cast to the container type.

    """
    klass = list
    _cast_types = (tuple,)

    # Describe the trait type
    info_text = 'an ordered list of NDArray trait'
    allow_none = True

    def __init__(self, default_value=None, **kwargs):

        super(ArrayList, self).__init__(trait=None, default_value=default_value,
                                    **kwargs)
        pass

    def validate_elements(self, obj, value):
        from spectrochempy.dataset import NDArray

        if value is None or len(value) == 0:
            # we allow none so this is possible
            return
        if not isinstance(value, NDArray):
            e = "The '%s' trait of %s instance must contains only NDArray's " \
                "or subclass of NDArray's," \
                " but a value of type `%s was specified." \
                % (self.name, class_of(obj), class_of(value))
            raise TraitError(e)
        value.sort()
        value = super(ArrayList, self).validate_elements(obj, value)
        return value

    def validate(self, obj, value):

        value = super(ArrayList, self).validate(object, value)
        value = self.validate_elements(obj, value)

        return value
