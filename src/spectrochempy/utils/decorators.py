# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import copy
import functools
import inspect
import re
from functools import partial
from functools import update_wrapper
from inspect import Parameter
from inspect import Signature
from inspect import signature
from typing import TypeVar
from warnings import warn

import traitlets as tr


def _deprecated_removal_sentence(removed=None, *, policy=False):
    """Return the removal sentence for a deprecation warning."""
    if policy:
        return (
            "It will not be removed before the SpectroChemPy deprecation policy "
            "is satisfied."
        )
    sremoved = f"version {removed}" if removed else "future version"
    return f"It will be removed in {sremoved}."


def format_deprecated_message(
    name,
    *,
    subject=None,
    kind="method",
    replace="",
    removed=None,
    extra_msg="",
    since=None,
    policy=False,
    action="is now deprecated",
):
    """Return a standardized deprecation message."""
    if subject is None:
        subject = f"The `{name}` {kind}" if kind else f"`{name}`"
    if since:
        msg = f"{subject} {action} since SpectroChemPy {since}. "
    else:
        msg = f"{subject} {action}. "

    if replace:
        msg += f"Use `{replace}` instead. "

    msg += _deprecated_removal_sentence(removed, policy=policy)
    if extra_msg:
        msg += f" {extra_msg.lstrip()}"
    return msg


def warn_deprecated(
    name,
    *,
    subject=None,
    kind="method",
    replace="",
    removed=None,
    extra_msg="",
    since=None,
    policy=False,
    action="is now deprecated",
    category=DeprecationWarning,
    stacklevel=2,
):
    """Emit a standardized ``DeprecationWarning``."""
    warn(
        format_deprecated_message(
            name,
            subject=subject,
            kind=kind,
            replace=replace,
            removed=removed,
            extra_msg=extra_msg,
            since=since,
            policy=policy,
            action=action,
        ),
        category=category,
        stacklevel=stacklevel,
    )


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


def deprecated(
    name=None,
    *,
    kind="method",
    replace="",
    removed=None,
    extra_msg="",
    policy=False,
):
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
    policy : bool, optional
        If True, express removal timing through the accepted SpectroChemPy
        deprecation policy instead of a version string.
    """

    if name is not None:
        kind = "attribute"
        warn_deprecated(
            name,
            kind=kind,
            replace=replace,
            removed=removed,
            extra_msg=extra_msg,
            policy=policy,
            stacklevel=2,
        )
        return None

    def deprecation_decorator(func):
        # @preserve_signature
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            if name.endswith("__init__"):
                name = name.split(".", maxsplit=1)[0]
            warn_deprecated(
                name,
                kind=kind,
                replace=replace,
                removed=removed,
                extra_msg=extra_msg,
                policy=policy,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


# ======================================================================================
# Useful decorators for Traitlets users.
# (modified from Traitlets : traitlets.signature_has_traits)
# ======================================================================================
T = TypeVar("T", bound=tr.HasTraits)

_KNOWN_NUMPYDOC_SECTIONS = {
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
_PARAMETER_DECLARATION_RE = re.compile(
    r"^\s*([*]{0,2}[A-Za-z_]\w*(?:\s*,\s*[*]{0,2}[A-Za-z_]\w*)*)\s*:\s*",
)


def _get_default(value):
    """Get default argument value, given the trait default value."""
    return Parameter.empty if value == tr.Undefined else value


def _parse_numpydoc_sections(docstring):
    """Return the summary block and the ordered numpydoc sections."""
    lines = docstring.splitlines()
    summary_lines = []
    sections = []
    current_section = None
    current_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if (
            stripped in _KNOWN_NUMPYDOC_SECTIONS
            and i + 1 < len(lines)
            and lines[i + 1].strip()
            and set(lines[i + 1].strip()) == {"-"}
        ):
            if current_section is None:
                summary_lines = current_lines
            else:
                sections.append((current_section, "\n".join(current_lines).rstrip()))

            current_section = stripped
            current_lines = []
            i += 2
            continue

        current_lines.append(line)
        i += 1

    if current_section is None:
        summary_lines = current_lines
    else:
        sections.append((current_section, "\n".join(current_lines).rstrip()))

    return "\n".join(summary_lines).strip(), sections


def _parse_parameter_entries(section_content):
    """
    Return free text and parameter entries from a numpydoc Parameters section body.

    Each entry keeps the declaration line and its whole indented description.
    """
    preamble_lines = []
    entries = []
    current_names = None
    current_lines = []

    def _flush_entry():
        if current_names is not None:
            entries.append((current_names, "\n".join(current_lines).rstrip()))

    for line in section_content.splitlines():
        match = _PARAMETER_DECLARATION_RE.match(line)
        if match:
            _flush_entry()
            current_names = tuple(name.strip() for name in match.group(1).split(","))
            current_lines = [line]
            continue

        if current_names is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    _flush_entry()
    return "\n".join(preamble_lines).strip("\n"), entries


def _format_numpydoc_section(name, content):
    """Format a numpydoc section block."""
    underline = "-" * len(name)
    content = content.rstrip()
    if content:
        return f"{name}\n{underline}\n{content}"
    return f"{name}\n{underline}"


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
    existing_doc = inspect.cleandoc(cls.__doc__ or "")

    trait_entry_map = {}
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

        lines = [f"{name} : {type_str}, optional, default: {default}"]
        desc = value.help or ""
        if desc:
            lines.extend(f"    {line}" for line in desc.splitlines())
        trait_entry_map[name] = "\n".join(lines)

    summary, sections = _parse_numpydoc_sections(existing_doc)
    params_sections = [content for name, content in sections if name == "Parameters"]

    params_preamble_parts = []
    manual_entries = []
    seen_param_names = set()

    for params_content in params_sections:
        params_preamble, parsed_entries = _parse_parameter_entries(params_content)
        if params_preamble:
            params_preamble_parts.append(params_preamble)

        for names, block in parsed_entries:
            if seen_param_names.intersection(names):
                continue
            manual_entries.append((names, block))
            seen_param_names.update(names)

    public_param_order = list(cls.__signature__.parameters)
    manual_entry_by_name = {}
    for entry_index, (names, _) in enumerate(manual_entries):
        for name in names:
            manual_entry_by_name.setdefault(name, entry_index)

    ordered_entries = []
    used_entry_indices = set()
    documented_names = set()

    for param_name in public_param_order:
        entry_index = manual_entry_by_name.get(param_name)
        if entry_index is not None:
            if entry_index not in used_entry_indices:
                names, block = manual_entries[entry_index]
                ordered_entries.append(block.rstrip())
                used_entry_indices.add(entry_index)
                documented_names.update(names)
            continue

        if param_name in trait_entry_map and param_name not in documented_names:
            ordered_entries.append(trait_entry_map[param_name])
            documented_names.add(param_name)

    for entry_index, (names, block) in enumerate(manual_entries):
        if entry_index not in used_entry_indices:
            ordered_entries.append(block.rstrip())
            documented_names.update(names)

    merged_params_parts = [part for part in params_preamble_parts if part.strip()]
    merged_params_parts.extend(entry for entry in ordered_entries if entry.strip())
    merged_params = "\n".join(merged_params_parts).strip()
    params_block = (
        _format_numpydoc_section("Parameters", merged_params) if merged_params else ""
    )

    doc_parts = []
    if summary:
        doc_parts.append(summary)

    inserted_params = False
    if not params_sections and params_block:
        doc_parts.append(params_block)
        inserted_params = True

    for section_name, content in sections:
        if section_name == "Parameters":
            if not inserted_params and params_block:
                doc_parts.append(params_block)
                inserted_params = True
            continue
        doc_parts.append(_format_numpydoc_section(section_name, content))

    doc = "\n\n".join(part for part in doc_parts if part)
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
        preserve_identity=False,
        use_snapshot=True,  # reuse the fit metadata snapshot on stored paths
        analysis_role=None,
    ):
        self.method = method
        update_wrapper(self, method)
        self.meta_from = meta_from
        self.units = units
        self.title = title
        self.typex = typex
        self.typey = typey
        self.typesingle = typesingle
        self.preserve_identity = preserve_identity
        self.use_snapshot = use_snapshot
        self.analysis_role = analysis_role

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

        # Identify the direct X and Y arguments of the call, if any.  Three
        # states are distinguished per role: no direct argument (the fitted
        # snapshot is reused), a direct NDDataset (its exact author is used),
        # and a direct array-like (no scientific provenance: the snapshot is
        # ignored and the runtime value is kept).  An explicit ``None`` counts
        # as "no direct argument" since it means "reuse the stored input".
        sentinel = object()
        direct_X = sentinel
        direct_Y = sentinel
        direct_X_kind = "none"
        direct_Y_kind = "none"
        x_params = ("X", "dataset", "X_transform")
        y_params = ("Y", "Y_transform")
        try:
            bound = signature(self.method).bind_partial(obj, *args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            for name, param in signature(self.method).parameters.items():
                if name == "self":
                    continue
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    for key, value in bound.arguments.get(name, {}).items():
                        if value is None:
                            continue
                        if key in x_params:
                            direct_X = value
                            direct_X_kind = (
                                "dataset"
                                if isinstance(value, NDDataset)
                                else "arraylike"
                            )
                        elif key in y_params:
                            direct_Y = value
                            direct_Y_kind = (
                                "dataset"
                                if isinstance(value, NDDataset)
                                else "arraylike"
                            )
                elif (
                    param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                    and name in bound.arguments
                ):
                    value = bound.arguments[name]
                    if value is None:
                        continue
                    if name in x_params:
                        direct_X = value
                        direct_X_kind = (
                            "dataset" if isinstance(value, NDDataset) else "arraylike"
                        )
                    elif name in y_params:
                        direct_Y = value
                        direct_Y_kind = (
                            "dataset" if isinstance(value, NDDataset) else "arraylike"
                        )
        else:
            # Fallback for exotic signatures: first two positional arguments
            # and the recognized keyword arguments.
            if args and args[0] is not None:
                direct_X = args[0]
                direct_X_kind = (
                    "dataset" if isinstance(args[0], NDDataset) else "arraylike"
                )
            if len(args) > 1 and args[1] is not None:
                direct_Y = args[1]
                direct_Y_kind = (
                    "dataset" if isinstance(args[1], NDDataset) else "arraylike"
                )
            for key in x_params:
                if kwargs.get(key) is not None:
                    direct_X = kwargs[key]
                    direct_X_kind = (
                        "dataset" if isinstance(kwargs[key], NDDataset) else "arraylike"
                    )
                    break
            for key in y_params:
                if kwargs.get(key) is not None:
                    direct_Y = kwargs[key]
                    direct_Y_kind = (
                        "dataset" if isinstance(kwargs[key], NDDataset) else "arraylike"
                    )
                    break

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
        preserve = self.preserve_identity and getattr(obj, "_preserve_identity", True)
        for idx, (data, meta_from) in enumerate(
            zip(data_tuple, meta_from_tuple, strict=False)
        ):
            X_transf = NDDataset(data)
            if isinstance(self.analysis_role, tuple):
                role_id = self.analysis_role[idx]
            else:
                role_id = self.analysis_role

            # Now set the NDDataset attributes from the original X

            # determine the input dataset of the current role
            X = getattr(obj, meta_from)
            direct_source = direct_X if meta_from == "_X" else direct_Y
            if direct_source is not sentinel and isinstance(direct_source, NDDataset):
                # A direct scientific source is authoritative for this call.
                metadata_source = direct_source
                use_snapshot = False
            else:
                metadata_source = X
                # The snapshot is reused only when no direct argument of this
                # role was given: a direct array-like carries no provenance
                # and must not leak the fitted source author.
                use_snapshot = self.use_snapshot and direct_source is sentinel

            # Promote 1D metadata source (e.g., _Y with 1D y) to 2D for
            # coordinate assignment so that dims[1], coord(1), shape[1] etc.
            # are valid.  The wrapped method always returns 2D outputs.
            if X.ndim == 1:
                import numpy as np

                dim0 = X.dims[0]
                coord0 = (
                    X.coordset[0].copy()
                    if X.coordset is not None and X.coordset[0] is not None
                    else None
                )
                X_new = NDDataset(np.empty((X.shape[0], 1)))
                X_new.dims = [dim0, "v"]
                X_new.set_coordset({dim0: coord0, "v": None})
                X_new.units = X.units
                X_new.title = X.title
                X_new.name = X.name
                X = X_new

            X_transf.meta = copy.deepcopy(metadata_source.meta)
            # The exact author of the scientific source is preferred over the
            # value recreated by the NDDataset coercion of the stored input.
            # A direct NDDataset argument is authoritative for the output of
            # its role (X or Y); a direct array-like argument carries no
            # scientific provenance and keeps the runtime value; stored
            # ``_X`` / ``_Y`` outputs reuse the metadata snapshot captured at
            # ``fit`` time only when no direct argument of that role was
            # given.  Supervised (multi-source) outputs such as ``predict``
            # do not use the snapshot (deferred to the multi-source policy,
            # PR 2).
            author = metadata_source.author
            if use_snapshot:
                source_metadata = getattr(obj, f"{meta_from}_source_metadata", None)
                if source_metadata is not None:
                    author = source_metadata.author
            X_transf.author = copy.copy(author)
            X_transf.description = copy.copy(metadata_source.description)
            X_transf.origin = copy.copy(metadata_source.origin)
            X_transf.filename = copy.copy(metadata_source.filename)
            if self.units is not None:
                if self.units == "keep":
                    X_transf.units = X.units
                else:
                    X_transf.units = self.units
            if preserve:
                X_transf.name = metadata_source.name
                X_transf._history = list(metadata_source._history or [])
            else:
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
                # Resolve component labels — defer to analysis subclasses
                # (e.g. PCA → PC1, PC2, …).
                def _component_labels(n):
                    if hasattr(obj, "_get_component_labels"):
                        return obj._get_component_labels(n)
                    return [f"#{i}" for i in range(n)]

                if self.typesingle == "components":
                    # occurs when the data are 1D such as ev_ratio...
                    X_transf.dims = ["k"]
                    X_transf.set_coordset(
                        k=Coord(
                            None,
                            labels=_component_labels(X_transf.shape[-1]),
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
                                labels=_component_labels(X_transf.shape[-1]),
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
                                labels=_component_labels(X_transf.shape[0]),
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
                                labels=_component_labels(X_transf.shape[-1]),
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

            if role_id is None or not hasattr(obj, "_apply_analysis_output_geometry"):
                # Preserve the legacy generic restoration path for non-analysis outputs.
                X_transf = obj._restore_masked_data(X_transf, axis=axis)

            # Only squeeze if the input was originally 1D (expanded to 2D)
            # This preserves intentionally 2D datasets with shape (1, N)
            if getattr(obj, "_X_original_ndim", 2) == 1:
                X_transf = X_transf.squeeze()
                if preserve:
                    X_transf._history = X_transf._history[:-1]

            if role_id is not None and hasattr(obj, "_apply_analysis_output_geometry"):
                X_transf = obj._apply_analysis_output_geometry(
                    X_transf,
                    role_id=role_id,
                    meta_from=meta_from,
                    direct_x=None if direct_X is sentinel else direct_X,
                    direct_y=None if direct_Y is sentinel else direct_Y,
                    direct_x_kind=direct_X_kind,
                    direct_y_kind=direct_Y_kind,
                )

            if role_id is not None and hasattr(obj, "_apply_analysis_output_metadata"):
                X_transf = obj._apply_analysis_output_metadata(
                    X_transf,
                    role_id=role_id,
                    meta_from=meta_from,
                    direct_x=None if direct_X is sentinel else direct_X,
                    direct_y=None if direct_Y is sentinel else direct_Y,
                    direct_x_kind=direct_X_kind,
                    direct_y_kind=direct_Y_kind,
                )
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
    preserve_identity=False,
    use_snapshot=True,
    analysis_role=None,
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
                preserve_identity=preserve_identity,
                use_snapshot=use_snapshot,
                analysis_role=analysis_role,
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
