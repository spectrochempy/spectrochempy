# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import copy
import functools
import inspect
from inspect import Parameter, Signature, signature
from textwrap import indent
from typing import Type, TypeVar
from warnings import warn

import traitlets as tr

from spectrochempy.utils.docstrings import _docstring


def preserve_signature(f):
    """
    A decorator for decorators, which preserves the signature of the function
    being wrapped. This preservation allows IDE function parameter hints to work
    on the wrapped function. To do this, the `__signature__` dunder is defined, or
    inherited, from the function being wrapped to the resulting wrapped function.

    Copied from
    https://github.com/PlasmaPy/PlasmaPy/blob/main/plasmapy/utils/decorators/helpers.py
    (PlasmaPy, LICENSE BSD-3)

    Parameters
    ----------
    f: callable
        The function being wrapped.

    Returns
    -------
    callable
        Wrapped version of the function.

    Examples
    --------
    >>> def a_decorator(f):
    ...     @preserve_signature
    ...     @functools.wraps(f)
    ...     def wrapper(*args, **kwargs):
    ...         return wrapper(*args, **kwargs)
    ...
    ...     return wrapper
    """
    # add '__signature__' if it does not exist
    # - this will preserve parameter hints in IDE's
    if not hasattr(f, "__signature__"):
        f.__signature__ = inspect.signature(f)

    return f


def deprecated(name=None, *, kind="method", replace="", removed=None, extra_msg=""):
    """
    Deprecation decorator.

    Parameters
    ----------
    name : str
        If name is specified, kind is mandatory set to attribute
        and the deprecated function is no more acting as a decorator.
    kind : str
        By default, it is method.
    replace : str, optional, default:None
        Name of the method that replace the deprecated one or None
    extra_msg : str
        Additional message.
    removed : str, optional
        Version string when this method will be removed
    """

    def output_warning_message(name, kind, replace, removed, extra_msg):
        sreplace = f"Use `{replace}` instead. " if replace is not None else ""
        msg = f" The `{name}` {kind} is now deprecated. {sreplace}"
        sremoved = f"version {removed}" if removed else "future version"
        msg += f"`{name}` {kind} will be removed in {sremoved}. "
        msg += extra_msg
        warn(
            msg,
            category=DeprecationWarning,
        )

    if name is not None:
        kind = "attribute"
        output_warning_message(name, kind, replace, removed, extra_msg)
        return

    def deprecation_decorator(func):
        # @preserve_signature
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            if name.endswith("__init__"):
                name = name.split(".", maxsplit=1)[0]
            output_warning_message(name, kind, replace, removed, extra_msg)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


# ======================================================================================
# Useful decorators for Traitlets users.
# (modified from Traitlets : traitlets.signature_has_traits)
# ======================================================================================
T = TypeVar("T", bound=tr.HasTraits)


def _get_default(value):
    """Get default argument value, given the trait default value."""
    return Parameter.empty if value == tr.Undefined else value


def signature_has_configurable_traits(cls: Type[T]) -> Type[T]:
    """
    Return a decorated class with a constructor signature that contain Trait names as kwargs.

    In addition, we update the corresponding docstring
    """
    traits = [
        (name, value)
        for name, value in cls.class_traits(config=True).items()
        if not name.startswith("_")
    ]

    # Taking the __init__ signature, as the cls signature is not initialized yet
    old_signature = signature(cls.__init__)
    old_parameter_names = list(old_signature.parameters)

    old_positional_parameters = []
    old_var_positional_parameter = (
        None  # This won't be None if the old signature contains *args
    )
    old_keyword_only_parameters = []
    old_var_keyword_parameter = (
        None  # This won't be None if the old signature contains **kwargs
    )

    for parameter_name in old_signature.parameters:
        # Copy the parameter
        parameter = copy.copy(old_signature.parameters[parameter_name])

        if (
            parameter.kind is Parameter.POSITIONAL_ONLY
            or parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        ):
            old_positional_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_POSITIONAL:
            old_var_positional_parameter = parameter

        elif parameter.kind is Parameter.KEYWORD_ONLY:
            old_keyword_only_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_KEYWORD:
            old_var_keyword_parameter = parameter

    # Unfortunately, if the old signature does not contain **kwargs, we can't do anything,
    # because it can't accept traits as keyword arguments
    if old_var_keyword_parameter is None:
        raise RuntimeError(
            "The {} constructor does not take **kwargs, which means that the signature "
            "can not be expanded with trait names".format(cls)
        )

    new_parameters = []

    # Append the old positional parameters (except `self` which is the first parameter)
    new_parameters += old_positional_parameters[1:]

    # Append *args if the old signature had it
    if old_var_positional_parameter is not None:
        new_parameters.append(old_var_positional_parameter)

    # Append the old keyword only parameters
    new_parameters += old_keyword_only_parameters

    # Append trait names as keyword only parameters in the signature
    new_parameters += [
        Parameter(
            name, kind=Parameter.KEYWORD_ONLY, default=_get_default(value.default_value)
        )
        for name, value in traits
        if name not in old_parameter_names
    ]

    # Append **kwargs  <- unlike traitlets we remove it
    # new_parameters.append(old_var_keyword_parameter)

    cls.__signature__ = Signature(new_parameters)  # type:ignore[attr-defined]

    # add the corresponding doctrings
    otherpar = ""
    for name, value in traits:
        # try to infer the parameter type
        type_ = type(value).__name__
        if type_ in ["Enum", "CaselessStrEnum"]:
            values = ""
            for val in value.values:
                values += f" ``'{val}'`` ,"
            type_ = f"any value of [{values.rstrip(',')}]"
        elif type_ == "Unicode":
            type_ = "`str`"
        elif type_ == "Any":
            type_ = "any value"
        elif type_ == "Union":
            type_ = value.info_text
        else:
            type_ = f"`{type_.lower()}`"

        default = value.default_value
        if isinstance(default, type(tr.Undefined)) or default is None:
            if type(value).__name__.lower() in ["tuple", "dict", "list"]:
                default = __builtins__[type(value).__name__.lower()]()
            else:
                default = "`None`"
        elif isinstance(default, str):
            default = f"``'{default}'``"
        otherpar += f"{name} : {type_}, optional, default: {default}\n"
        desc = f"{value.help}\n"
        desc = indent(desc, "    ")
        otherpar += desc

    doc = _docstring.dedent(cls.__doc__)
    _docstring.get_full_description(doc, base=cls.__name__)
    _docstring.get_sections(
        doc,
        base=cls.__name__,
        sections=[
            "Parameters",
            "Other Parameters",
            "See Also",
            "Examples",
            "Notes",
            "References",
        ],
    )
    _docstring.params[f"{cls.__name__}.parameters"] += f"\n{otherpar.strip()}"
    doc = "\n" + _docstring.params[f"{cls.__name__}.full_desc"]
    doc += "\n\n"
    doc += "Parameters\n"
    doc += "----------\n"
    doc += _docstring.params[f"{cls.__name__}.parameters"]
    doc += "\n"
    if _docstring.params[f"{cls.__name__}.other_parameters"]:
        doc += "\nOther Parameters\n"
        doc += "----------------\n"
        doc += _docstring.params[f"{cls.__name__}.other_parameters"]
        doc += "\n"
    if _docstring.params[f"{cls.__name__}.see_also"]:
        doc += "\nSee Also\n"
        doc += "--------\n"
        doc += _docstring.params[f"{cls.__name__}.see_also"]
        doc += "\n"
    if _docstring.params[f"{cls.__name__}.references"]:
        doc += "\nReferences\n"
        doc += "----------\n"
        doc += _docstring.params[f"{cls.__name__}.references"]
        doc += "\n"
    if _docstring.params[f"{cls.__name__}.examples"]:
        doc += "\nExamples\n"
        doc += "--------\n"
        doc += _docstring.params[f"{cls.__name__}.examples"]
        doc += "\n"
    if _docstring.params[f"{cls.__name__}.notes"]:
        doc += "\nNotes\n"
        doc += "-----\n"
        doc += _docstring.params[f"{cls.__name__}.notes"]
        doc += "\n"
    cls.__doc__ = doc

    # some attribute doc
    if hasattr(cls, "config"):
        cls.config.__doc__ = "`traitlets.config.Config` object."
        cls.parent.__doc__ = None
    return cls
