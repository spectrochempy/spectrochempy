"""Deterministic Markdown reporting for explicit scientific inputs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from urllib.parse import quote
from urllib.parse import urlparse

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import Unit

__all__ = ["FigureReference", "render_dataset_markdown"]


_SCOPE_MESSAGE = (
    "This report summarizes recorded dataset information and processing history. "
    "It is not an executable workflow or a complete provenance record."
)
_MAX_COLLECTION_ITEMS = 32
_MAX_NESTING_DEPTH = 4
_MAX_TEXT_LENGTH = 4096
_MISSING = "[absent]"


@dataclass(frozen=True)
class FigureReference:
    """
    An explicit, report-relative or HTTP(S) figure reference.

    The caller supplies every reference; the renderer never captures Matplotlib
    state.

    Parameters
    ----------
    label : str
        Figure label, used for the section heading and the image alternative
        text.
    target : str
        HTTP(S) URL or relative POSIX path, resolved relative to the exported
        Markdown file. Spaces and Markdown delimiters are URL-encoded.
        Other URL schemes and absolute paths are rejected. The target is not
        checked for existence and is never copied.
    caption : str, optional
        Optional caption rendered after the image.

    See Also
    --------
    reporting.render_dataset_markdown : Render a dataset report.
    NDDataset.to_markdown_report : Shortcut on a dataset instance.

    Examples
    --------
    .. code-block:: python

       from spectrochempy.reporting import FigureReference

       figure = FigureReference(
           "Spectrum",
           "figures/spectrum.png",
           "Measured signal.",
       )
    """

    label: str
    target: str
    caption: str | None = None


def render_dataset_markdown(
    dataset: NDDataset,
    *,
    title: str,
    metadata: Sequence[str] = (),
    figures: Sequence[FigureReference] = (),
) -> str:
    """
    Render one :class:`NDDataset` as a deterministic Markdown report.

    Rendering is deterministic, does not modify the dataset, and never captures
    Matplotlib state.

    Parameters
    ----------
    dataset : NDDataset
        Dataset to describe. The renderer does not modify it.
    title : str
        Required document title.
    metadata : sequence of str, optional
        Explicit metadata keys to include, in report order. This allowlist is
        the only metadata selection mechanism; no key is discovered
        automatically.
    figures : sequence of FigureReference, optional
        Explicit figure references. Matplotlib figures and implicit capture are
        not supported.

    Returns
    -------
    str
        Deterministic Markdown document ending with a single newline.

    See Also
    --------
    NDDataset.to_markdown_report : Shortcut on a dataset instance.
    reporting.FigureReference : Explicit figure reference.

    Notes
    -----
    Free text is rendered as supplied and escaped for Markdown. The renderer
    does not detect confidential information or redact paths. Callers are
    responsible for the metadata, origin, history and figure references they
    include. Select metadata keys and sanitize sensitive content before sharing.
    Every recorded history entry is rendered in its original order. Individual
    values may be explicitly truncated for length, collection size or depth.

    Examples
    --------
    .. code-block:: python

       import spectrochempy as scp
       from spectrochempy.reporting import FigureReference

       dataset = scp.NDDataset([1.0, 2.0, 3.0], title="Absorbance")
       dataset.meta.operator = "Ada"
       report = dataset.to_markdown_report(
           title="IR experiment",
           metadata=("operator",),
           figures=(FigureReference("Spectrum", "figures/spectrum.png"),),
       )
       print(report)
    """
    if not isinstance(dataset, NDDataset):
        raise TypeError("dataset must be an NDDataset")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if isinstance(metadata, str):
        raise TypeError("metadata must be a sequence of strings")
    metadata = tuple(metadata)
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("metadata keys must be strings")
    figures = tuple(figures)
    _validate_figures(figures)

    lines = [
        f"# {_text(title)}",
        "",
        "## Datasets",
        "",
        f"### {_dataset_heading(dataset)}",
        "",
    ]
    lines.extend(_dataset_summary(dataset))
    lines.extend(_coordinates(dataset))
    lines.extend(_metadata(dataset, metadata))
    lines.extend(_history(dataset))
    lines.extend(_figures(figures))
    lines.extend(["", _SCOPE_MESSAGE])
    return "\n".join(lines).rstrip() + "\n"


def _dataset_heading(dataset: NDDataset) -> str:
    if dataset.has_defined_name:
        return _text(dataset.name)
    return "Dataset"


def _dataset_summary(dataset: NDDataset) -> list[str]:
    lines = ["- Shape: " + _code(repr(dataset.shape))]
    lines.append("- Dimensions: " + _code(", ".join(dataset.dims)))
    lines.append("- Data unit: " + _unit(dataset.units))
    if dataset.title != "<untitled>":
        lines.append("- Title: " + _text(dataset.title))
    origin = dataset.origin
    if isinstance(origin, str) and not origin.strip():
        origin = None
    lines.append("- Origin: " + _value(origin))
    return lines


def _coordinates(dataset: NDDataset) -> list[str]:
    lines = ["", "#### Coordinates", ""]
    coordset = dataset.coordset
    if coordset is None:
        return [*lines, "- No coordinates provided."]

    names = coordset.names
    for dim in dataset.dims:
        coord = dataset.coord(dim) if dim in names else None
        if coord is None:
            lines.append(f"- `{_escape(dim)}`: —")
            continue
        if hasattr(coord, "default"):
            coord = coord.default
        if coord is None:
            lines.append(f"- `{_escape(dim)}`: —")
            continue
        label = coord.title if coord.title != "<untitled>" else coord.name
        summary = [
            f"{_code(dim)} — {_text(label)}",
            f"size: {coord.size}",
            f"unit: {_unit(coord.units)}",
        ]
        extent = _coordinate_extent(coord)
        if extent is not None:
            summary.append(extent)
        lines.append("- " + "; ".join(summary))
    return lines


def _coordinate_extent(coord) -> str | None:
    data = coord.data
    if data is None or len(data) == 0:
        return None
    array = np.ma.asarray(data)
    if array.dtype.kind in "iuf":
        valid = np.ma.masked_invalid(array).compressed()
        if valid.size == 0:
            return None
        return f"min {_value(valid.min())}; max {_value(valid.max())}"
    return f"first {_value(data[0])}; last {_value(data[-1])}"


def _metadata(dataset: NDDataset, keys: Sequence[str]) -> list[str]:
    if not keys:
        return []
    lines = ["", "#### Selected metadata", ""]
    for key in keys:
        if key not in dataset.meta:
            lines.append(f"- {_code(key)}: {_MISSING}")
            continue
        lines.append(f"- {_code(key)}: {_value(dataset.meta[key])}")
    return lines


def _history(dataset: NDDataset) -> list[str]:
    lines = ["", "#### Processing history", ""]
    entries = dataset.history_entries
    if not entries:
        return [*lines, "- No history entry provided."]

    for index, entry in enumerate(entries, start=1):
        operation = entry["operation"] if entry["operation"] is not None else "text"
        lines.extend([f"##### {index}. {_history_label(operation)}", ""])
        parameters = entry["parameters"]
        if isinstance(parameters, Mapping):
            if "model" in parameters:
                model = parameters["model"]
                method = _code(model) if isinstance(model, str) else _value(model)
                lines.append(f"- Method: {method}")
            remaining = {
                key: value
                for key, value in parameters.items()
                if key not in {"model", "sources"}
            }
            if remaining:
                lines.append("- Parameters:")
                lines.extend(_history_value(remaining))
            else:
                lines.append("- Parameters: —")
            if "sources" in parameters:
                lines.append("- Sources:")
                lines.extend(_history_value(parameters["sources"]))
        else:
            lines.append(f"- Parameters: {_value(parameters)}")
        lines.extend(
            [
                "- Details:",
                f"  - Timestamp: {_code(_datetime(entry['date']))}",
                f"  - Operation: {_code(operation)}",
                f"  - Message: {_value(entry['message'])}",
                "",
            ]
        )
    return lines[:-1]


def _history_value(value, *, depth: int = 0, indent: str = "  ") -> list[str]:
    if not isinstance(value, Mapping | list | tuple):
        return [f"{indent}- {_value(value)}"]
    if depth >= _MAX_NESTING_DEPTH:
        return [f"{indent}- [truncated: maximum depth]"]
    if not value:
        return [f"{indent}- {'{}' if isinstance(value, Mapping) else '[]'}"]
    items = (
        [(key, value[key]) for key in sorted(value, key=str)[:_MAX_COLLECTION_ITEMS]]
        if isinstance(value, Mapping)
        else [
            (f"Item {i}", item)
            for i, item in enumerate(value[:_MAX_COLLECTION_ITEMS], start=1)
        ]
    )
    lines = []
    for key, child in items:
        label = _history_label(key)
        if isinstance(child, Mapping | list | tuple):
            lines.append(f"{indent}- {label}:")
            lines.extend(_history_value(child, depth=depth + 1, indent=indent + "  "))
        else:
            lines.append(f"{indent}- {label}: {_value(child)}")
    if len(value) > _MAX_COLLECTION_ITEMS:
        lines.append(
            f"{indent}- [truncated: {len(value) - _MAX_COLLECTION_ITEMS} more items]"
        )
    return lines


def _history_label(value) -> str:
    label = str(value).replace("_", " ")
    return _text(label[:1].upper() + label[1:])


def _figures(figures: Sequence[FigureReference]) -> list[str]:
    if not figures:
        return []
    lines = ["", "## Figures", ""]
    for figure in figures:
        target = _figure_target(figure.target)
        label = _text(figure.label)
        lines.append(f"### {label}")
        lines.extend(["", f"![{label}]({target})"])
        if figure.caption is not None:
            lines.extend(["", f"Caption: {_text(figure.caption)}"])
        lines.append("")
    return lines[:-1]


def _validate_figures(figures: Sequence[FigureReference]) -> None:
    for figure in figures:
        if not isinstance(figure, FigureReference):
            raise TypeError("figures must contain FigureReference instances")
        if not isinstance(figure.label, str) or not figure.label:
            raise ValueError("figure label must be a non-empty string")
        if figure.caption is not None and not isinstance(figure.caption, str):
            raise TypeError("figure caption must be a string or None")
        _figure_target(figure.target)


def _figure_target(target: str) -> str:
    if not isinstance(target, str) or not target:
        raise ValueError("figure target must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in target):
        raise ValueError("figure target must not contain control characters")
    try:
        parsed = urlparse(target)
    except ValueError as exc:
        raise ValueError("figure target must be a valid URL or relative path") from exc
    if parsed.scheme:
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("figure target must be an HTTP(S) URL or relative path")
    elif target.startswith(("/", "\\")):
        raise ValueError("figure target must be an HTTP(S) URL or relative path")
    if parsed.scheme:
        return parsed._replace(
            netloc=quote(parsed.netloc, safe="[]:@%"),
            path=quote(parsed.path, safe="/:%+@,;~!$&*-"),
            params=quote(parsed.params, safe="=:%+@,;~!$&*-"),
            query=quote(parsed.query, safe="/?:=+@,;~!$&*-%"),
            fragment=quote(parsed.fragment, safe="/?:=+@,;~!$&*-%"),
        ).geturl()
    return quote(target, safe="/%+@,~!$*-")


def _datetime(value: datetime) -> str:
    return value.isoformat()


def _unit(value) -> str:
    return "—" if value is None else _code(str(value))


def _value(value, *, depth: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, str | Path):
        return _text(str(value))
    if isinstance(value, bool | Integral | np.integer):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, np.floating):
        return _numpy_float(value)
    if isinstance(value, datetime):
        return _datetime(value)
    if isinstance(value, Unit):
        return _unit(value)
    if isinstance(value, Quantity):
        return f"{_value(value.magnitude, depth=depth + 1)} {_unit(value.units)}"
    if isinstance(value, Mapping | list | tuple):
        if depth >= _MAX_NESTING_DEPTH:
            return "[truncated: maximum depth]"
        if isinstance(value, Mapping):
            items = [
                f"{_text(str(key))}: {_value(value[key], depth=depth + 1)}"
                for key in sorted(value, key=str)[:_MAX_COLLECTION_ITEMS]
            ]
            opening, closing = "{", "}"
        else:
            items = [
                _value(item, depth=depth + 1) for item in value[:_MAX_COLLECTION_ITEMS]
            ]
            opening, closing = "[", "]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            items.append(
                f"[truncated: {len(value) - _MAX_COLLECTION_ITEMS} more items]"
            )
        return opening + ", ".join(items) + closing
    return f"[omitted: {type(value).__name__}]"


def _float(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+∞" if value > 0 else "-∞"
    return repr(float(value))


def _numpy_float(value: np.floating) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isinf(value):
        return "+∞" if value > 0 else "-∞"
    return str(value)


def _text(value: str) -> str:
    if len(value) > _MAX_TEXT_LENGTH:
        return _escape(value[:_MAX_TEXT_LENGTH]) + " [truncated: text too long]"
    return _escape(value)


def _escape(value: str) -> str:
    value = value.replace("\\", "\\\\")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", "\\n")
    for character in "`*[]<>#~":
        value = value.replace(character, f"\\{character}")
    return re.sub(r"(?<!\w)_|_(?!\w)", r"\\_", value)


def _code(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    delimiter = "`"
    while delimiter in value:
        delimiter += "`"
    if not value or value.startswith("`") or value.endswith("`"):
        value = f" {value} "
    return f"{delimiter}{value}{delimiter}"
