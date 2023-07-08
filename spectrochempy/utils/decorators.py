# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import copy
import functools
import inspect
from functools import partial, update_wrapper
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
            # print(name, value, type_)
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


# ======================================================================================
# A decorator to transform np.ndarray output from models to NDDataset
# according to the X (default) and/or Y input
# ======================================================================================
class _set_output(object):
    def __init__(
        self,
        method,
        *args,
        meta_from="_X",  # the attribute or tuple of attributes from which meta data are taken
        units="keep",
        title="keep",
        typex=None,
        typey=None,
        typesingle=None,
    ):
        self.method = method
        update_wrapper(self, method)
        self.meta_from = meta_from
        self.units = units
        self.title = title
        self.typex = typex
        self.typey = typey
        self.typesingle = typesingle

    @preserve_signature
    def __get__(self, obj, objtype):
        """Support instance methods."""
        newfunc = partial(self.__call__, obj)
        update_wrapper(newfunc, self.method)
        return newfunc

    def __call__(self, obj, *args, **kwargs):
        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        # HACK to be able to used deprecated alias of the method, without error
        # because if not this modification obj appears two times
        if args and type(args[0]) == type(obj):
            args = args[1:]

        # get the method output - one or two arrays depending on the method and *args
        output = self.method(obj, *args, **kwargs)

        # restore eventually masked rows and columns
        axis = "both"
        if self.typex is not None and self.typex != "features":
            axis = 0
        elif self.typey is not None:
            axis = 1

        # if a single array was returned...
        if not isinstance(output, tuple):
            # ... make a tuple of 1 array:
            data_tuple = (output,)
            # ... and a tuple of 1 from_meta element:
            if not isinstance(self.meta_from, tuple):
                meta_from_tuple = (self.meta_from,)
            else:
                # ensure that the first one
                meta_from_tuple = (self.meta_from[0],)
        else:
            data_tuple = output
            meta_from_tuple = self.meta_from

        out = []
        for data, meta_from in zip(data_tuple, meta_from_tuple):
            X_transf = NDDataset(data)

            # Now set the NDDataset attributes from the original X

            # determine the input X dataset
            X = getattr(obj, meta_from)

            if self.units is not None:
                if self.units == "keep":
                    X_transf.units = X.units
                else:
                    X_transf.units = self.units
            X_transf.name = f"{X.name}_{obj.name}.{self.method.__name__}"
            X_transf.history = f"Created using method {obj.name}.{self.method.__name__}"
            if self.title is not None:
                if self.title == "keep":
                    X_transf.title = X.title
                else:
                    X_transf.title = self.title
            # make coordset
            M, N = X.shape

            if X_transf.shape == X.shape and self.typex is None and self.typey is None:
                X_transf.dims = X.dims
                X_transf.set_coordset({X.dims[0]: X.coord(0), X.dims[1]: X.coord(1)})
            else:
                if self.typey == "components":
                    X_transf.dims = ["k", X.dims[1]]
                    X_transf.set_coordset(
                        {
                            "k": Coord(
                                None,
                                labels=["#%d" % (i) for i in range(X_transf.shape[0])],
                                title="components",
                            ),
                            X.dims[1]: X.coord(1).copy()
                            if X.coord(-1) is not None
                            else None,
                        }
                    )
                if self.typex == "components":
                    X_transf.dims = [X.dims[0], "k"]
                    X_transf.set_coordset(
                        {
                            X.dims[0]: X.coord(0).copy()
                            if X.coord(0) is not None
                            else None,
                            # cannot use X.y in case of transposed X
                            "k": Coord(
                                None,
                                labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                        }
                    )
                if self.typex == "features":
                    X_transf.dims = ["k", X.dims[1]]
                    X_transf.set_coordset(
                        {
                            "k": Coord(
                                None,
                                labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                            X.dims[1]: X.coord(1).copy()
                            if X.coord(1) is not None
                            else None,
                        }
                    )
                if self.typey == "features":
                    X_transf.dims = [X.dims[1], "k"]
                    X_transf.set_coordset(
                        {
                            X.dims[1]: X.coord(1).copy()
                            if X.coord(1) is not None
                            else None,
                            "k": Coord(
                                None,
                                labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                        }
                    )
                if self.typesingle == "components":
                    # occurs when the data are 1D such as ev_ratio...
                    X_transf.dims = ["k"]
                    X_transf.set_coordset(
                        k=Coord(
                            None,
                            labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                            title="components",
                        ),
                    )
                if self.typesingle == "targets":
                    # occurs when the data are 1D such as PLSRegression intercept...
                    if X.coordset[0].labels is not None:
                        labels = X.coordset[0].labels
                    else:
                        labels = ["#%d" % (i + 1) for i in range(X.shape[-1])]
                    X_transf.dims = ["j"]
                    X_transf.set_coordset(
                        j=Coord(
                            None,
                            labels=labels,
                            title="targets",
                        ),
                    )

            # eventually restore masks
            X_transf = obj._restore_masked_data(X_transf, axis=axis)
            out.append(X_transf.squeeze())

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


def _wrap_ndarray_output_to_nddataset(
    method=None,
    meta_from="_X",
    units="keep",
    title="keep",
    typex=None,
    typey=None,
    typesingle=None,
):
    # wrap _set_output to allow for deferred calling
    if method:
        # case of the decorator without argument
        out = _set_output(method)
    else:
        # and with argument
        def wrapper(method):
            return _set_output(
                method,
                meta_from=meta_from,
                units=units,
                title=title,
                typex=typex,
                typey=typey,
                typesingle=typesingle,
            )

        out = wrapper
    return out


# ======================================================================================
def _units_agnostic_method(method):
    @functools.wraps(method)
    def wrapper(dataset, **kwargs):
        # On which axis do we want to shift (get axis from arguments)
        axis, dim = dataset.get_axis(**kwargs, negative_axis=True)

        # output dataset inplace (by default) or not
        if not kwargs.pop("inplace", False):
            new = dataset.copy()  # copy to be sure not to modify this dataset
        else:
            new = dataset

        swapped = False
        if axis != -1:
            new.swapdims(axis, -1, inplace=True)  # must be done in  place
            swapped = True

        data = method(new.data, **kwargs)
        new._data = data

        new.history = (
            f"`{method.__name__}` shift performed on dimension "
            f"`{dim}` with parameters: {kwargs}"
        )

        # restore original data order if it was swapped
        if swapped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace

        return new

    return wrapper
