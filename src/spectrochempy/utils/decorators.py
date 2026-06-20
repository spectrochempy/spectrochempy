# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import copy
import functools
import inspect
from functools import partial
from functools import update_wrapper
from inspect import Parameter
from inspect import Signature
from inspect import signature
from typing import TypeVar
from warnings import warn

import traitlets as tr


def preserve_signature(f):
    """
    Preserve the signature of the function being wrapped.

    This preservation allows IDE function parameter hints to work
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
    Deprecate a function or attribute.

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
            stacklevel=2,
        )

    if name is not None:
        kind = "attribute"
        output_warning_message(name, kind, replace, removed, extra_msg)
        return None

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


def signature_has_configurable_traits(cls: type[T]) -> type[T]:
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
            f"The {cls} constructor does not take **kwargs, which means that the signature "
            "can not be expanded with trait names",
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
            name,
            kind=Parameter.KEYWORD_ONLY,
            default=_get_default(value.default_value),
        )
        for name, value in traits
        if name not in old_parameter_names
    ]

    # Append **kwargs  <- unlike traitlets we remove it
    # new_parameters.append(old_var_keyword_parameter)

    cls.__signature__ = Signature(new_parameters)  # type:ignore[attr-defined]

    # Build docstring from traits and existing docstring
    # -------------------------------------------------
    # Start with the existing docstring (summary + extended summary)
    existing_doc = cls.__doc__ or ""

    # Build the Parameters section from traits
    trait_params = ""
    trait_names = {name for name, _ in traits}

    for name, value in traits:
        # Determine type string
        type_ = type(value).__name__
        if type_ in ["Enum", "CaselessStrEnum"]:
            values = ", ".join(f"``'{val}'``" for val in value.values)
            type_str = f"any value of [{values}]"
        elif type_ == "Unicode":
            type_str = "`str`"
        elif type_ == "Any":
            type_str = "any value"
        elif type_ == "Union":
            type_str = value.info_text
        else:
            type_str = f"`{type_.lower()}`"

        # Determine default
        default = value.default_value
        if isinstance(default, type(tr.Undefined)) or default is None:
            if type_.lower() in ["tuple", "dict", "list"]:
                default = repr(__builtins__[type_.lower()]())
            else:
                default = "`None`"
        elif isinstance(default, str):
            default = f"``'{default}'``"
        elif isinstance(default, bool):
            default = f"``{default}``"
        else:
            default = f"``{default!r}``"

        trait_params += f"{name} : {type_str}, optional, default: {default}\n"
        desc = value.help or ""
        if desc:
            for line in desc.splitlines():
                trait_params += f"    {line}\n"
        else:
            trait_params += "\n"

    # Parse docstring into sections using a robust line-by-line approach
    # Known numpydoc section names
    KNOWN_SECTIONS = {
        "Parameters",
        "Returns",
        "Yields",
        "Other Parameters",
        "Raises",
        "Warns",
        "Warnings",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Attributes",
        "Methods",
    }

    lines = existing_doc.split("\n")
    sections_dict = {}
    current_section = "__summary__"
    current_content = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a section header
        if stripped in KNOWN_SECTIONS and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is all dashes (allowing whitespace)
            if next_line.strip() and all(c == "-" for c in next_line.strip()):
                # Save current section
                if current_content:
                    content = "\n".join(current_content).rstrip()
                    if current_section in sections_dict:
                        sections_dict[current_section] += "\n" + content
                    else:
                        sections_dict[current_section] = content

                # Start new section
                current_section = stripped
                current_content = []
                i += 2  # Skip header and underline
                continue

        current_content.append(line)
        i += 1

    # Save final section
    if current_content:
        content = "\n".join(current_content).rstrip()
        if current_section in sections_dict:
            sections_dict[current_section] += "\n" + content
        else:
            sections_dict[current_section] = content

    # Extract summary
    summary = sections_dict.pop("__summary__", "").strip()

    # Extract all Parameters content and merge
    params_content = ""
    if "Parameters" in sections_dict:
        params_content = sections_dict.pop("Parameters").strip()

    # Build merged Parameters section
    merged_params = []

    # Add existing non-trait params
    if params_content:
        for line in params_content.split("\n"):
            stripped = line.strip()
            if stripped:
                # Check if it's a parameter definition (param_name : ...)
                import re

                param_match = re.match(r"^(\w+)\s*:", stripped)
                if param_match:
                    param_name = param_match.group(1)
                    if param_name not in trait_names:
                        merged_params.append(line)
                else:
                    # It's a continuation line
                    merged_params.append(line)
            else:
                merged_params.append(line)

    # Add trait params
    if trait_params:
        merged_params.append(trait_params.rstrip())

    # Build final docstring with proper section order
    doc_parts = []

    # Add summary
    if summary:
        doc_parts.append(summary)

    # Add merged Parameters section
    if merged_params:
        params_section = "Parameters\n----------\n" + "\n".join(merged_params)
        doc_parts.append(params_section)

    # Add remaining sections in preferred order
    section_order = [
        "Returns",
        "Yields",
        "Other Parameters",
        "Raises",
        "Warns",
        "Warnings",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Attributes",
        "Methods",
    ]

    for section_name in section_order:
        if section_name in sections_dict:
            content = sections_dict[section_name]
            underline = "-" * len(section_name)
            doc_parts.append(f"{section_name}\n{underline}\n{content}")

    # Join with double newlines
    doc = "\n\n".join(doc_parts)
    # Add leading newline to satisfy numpydoc GL01 (expects 1 blank line at start)
    # and trailing newline for GL02 (expects 1 blank line at end)
    cls.__doc__ = "\n" + doc + "\n"

    # some attribute doc
    if hasattr(cls, "config"):
        cls.config.__doc__ = "`traitlets.config.Config` object."
        cls.parent.__doc__ = None
    return cls


# ======================================================================================
# A decorator to transform np.ndarray output from models to NDDataset
# according to the X (default) and/or Y input
# ======================================================================================
class _set_output:
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
        if args and type(args[0]) is type(obj):
            args = args[1:]

        original_X = None
        if args and isinstance(args[0], NDDataset):
            original_X = args[0]
        elif isinstance(kwargs.get("dataset"), NDDataset):
            original_X = kwargs["dataset"]

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
        for data, meta_from in zip(data_tuple, meta_from_tuple, strict=False):
            X_transf = NDDataset(data)

            # Now set the NDDataset attributes from the original X

            # determine the input X dataset
            X = getattr(obj, meta_from)
            metadata_source = (
                original_X if meta_from == "_X" and original_X is not None else X
            )

            X_transf.meta = copy.deepcopy(metadata_source.meta)
            X_transf.author = copy.copy(metadata_source.author)
            X_transf.description = copy.copy(metadata_source.description)
            X_transf.origin = copy.copy(metadata_source.origin)
            X_transf.filename = copy.copy(metadata_source.filename)
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
                # Allow a processing method to annotate the output title (e.g. to
                # flag a derived quantity) by exposing ``_output_title_suffix``
                # on the instance, without having to override this decorator.
                suffix = getattr(obj, "_output_title_suffix", None)
                if suffix and X_transf.title:
                    X_transf.title = f"{X_transf.title} {suffix}"
            # make coordset
            M, N = X.shape

            if X_transf.shape == X.shape and self.typex is None and self.typey is None:
                X_transf.dims = X.dims
                X_transf.set_coordset({X.dims[0]: X.coord(0), X.dims[1]: X.coord(1)})
            else:
                if self.typesingle == "components":
                    # occurs when the data are 1D such as ev_ratio...
                    X_transf.dims = ["k"]
                    X_transf.set_coordset(
                        k=Coord(
                            None,
                            labels=[f"#{i}" for i in range(X_transf.shape[-1])],
                            title="components",
                        ),
                    )
                elif self.typesingle == "targets":
                    # occurs when the data are 1D such as PLSRegression intercept...
                    if X.coordset[0].labels is not None:
                        labels = X.coordset[0].labels
                    else:
                        labels = [f"#{i + 1}" for i in range(X.shape[-1])]
                    X_transf.dims = ["j"]
                    X_transf.set_coordset(
                        j=Coord(
                            None,
                            labels=labels,
                            title="targets",
                        ),
                    )
                elif self.typey == "features" and self.typex == "components":
                    # combined: dim[0]=features, dim[1]=components
                    X_transf.dims = [X.dims[1], "k"]
                    X_transf.set_coordset(
                        {
                            X.dims[1]: (
                                X.coord(1).copy() if X.coord(1) is not None else None
                            ),
                            "k": Coord(
                                None,
                                labels=[f"#{i}" for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                        },
                    )
                elif self.typey == "components":
                    X_transf.dims = ["k", X.dims[1]]
                    X_transf.set_coordset(
                        {
                            "k": Coord(
                                None,
                                labels=[f"#{i}" for i in range(X_transf.shape[0])],
                                title="components",
                            ),
                            X.dims[1]: (
                                X.coord(1).copy() if X.coord(-1) is not None else None
                            ),
                        },
                    )
                elif self.typex == "components":
                    X_transf.dims = [X.dims[0], "k"]
                    X_transf.set_coordset(
                        {
                            X.dims[0]: (
                                X.coord(0).copy() if X.coord(0) is not None else None
                            ),
                            # cannot use X.y in case of transposed X
                            "k": Coord(
                                None,
                                labels=[f"#{i}" for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                        },
                    )
                elif self.typex == "features":
                    X_transf.dims = ["k", X.dims[1]]
                    X_transf.set_coordset(
                        {
                            "k": Coord(
                                None,
                                labels=[f"#{i}" for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                            X.dims[1]: (
                                X.coord(1).copy() if X.coord(1) is not None else None
                            ),
                        },
                    )
                elif self.typey == "features":
                    X_transf.dims = [X.dims[1], "k"]
                    X_transf.set_coordset(
                        {
                            X.dims[1]: (
                                X.coord(1).copy() if X.coord(1) is not None else None
                            ),
                            "k": Coord(
                                None,
                                labels=[f"#{i}" for i in range(X_transf.shape[-1])],
                                title="components",
                            ),
                        },
                    )

            # eventually restore masks
            X_transf = obj._restore_masked_data(X_transf, axis=axis)
            # Only squeeze if the input was originally 1D (expanded to 2D)
            # This preserves intentionally 2D datasets with shape (1, N)
            if getattr(obj, "_X_original_ndim", 2) == 1:
                out.append(X_transf.squeeze())
            else:
                out.append(X_transf)

        if len(out) == 1:
            return out[0]
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
        new = dataset.copy() if not kwargs.pop("inplace", False) else dataset

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
