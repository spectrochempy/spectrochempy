"""Deterministic Markdown reporting for explicit scientific inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import unquote
from urllib.parse import urlparse

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import Unit

__all__ = ["FigureReference", "render_dataset_markdown"]


_SCOPE_MESSAGE = (
    "This report summarizes the supplied objects and history. It is not an "
    "executable recipe or a proof of exhaustive provenance. It renders supplied "
    "text with Markdown escaping but does not detect or certify that it is free "
    "of sensitive information; select and sanitize metadata, origin and history "
    "before rendering."
)
_MAX_COLLECTION_ITEMS = 32
_MAX_NESTING_DEPTH = 4
_MAX_TEXT_LENGTH = 4096
_MAX_RENDERED_NODES = 256
_MAX_RENDERED_CHARS = 16384
_RENDER_BUDGET = "[omitted: render budget exhausted]"
_MISSING = "[absent]"
_REDACTED_PATH = "[redacted: path]"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class FigureReference:
    """
    An explicit, report-relative or HTTPS figure reference.

    The caller supplies every reference; the renderer never captures Matplotlib
    state.

    Parameters
    ----------
    label : str
        Figure label, used for the section heading and the image alternative
        text.
    target : str
        HTTP(S) URL or safe relative POSIX path. Whitespace, control
        characters, Markdown syntax characters and invalid percent escapes are
        rejected. The target is not checked for existence and is never copied.
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
    does not guess which strings are sensitive: only :class:`pathlib.Path`
    values are redacted. Select safe metadata keys and sanitize titles, origin,
    history and figure labels or captions before rendering.

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
        "## Scope and limitations",
        "",
        f"> {_SCOPE_MESSAGE}",
    ]
    lines.extend(["", "## Datasets", "", f"### {_dataset_heading(dataset)}", ""])
    lines.extend(_dataset_summary(dataset))
    lines.extend(_coordinates(dataset))
    lines.extend(_metadata(dataset, metadata))
    lines.extend(_history(dataset))
    lines.extend(_figures(figures))
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
    lines = ["", "#### History", ""]
    entries = dataset.history_entries
    if not entries:
        return [*lines, "- No history entry provided."]

    for index, entry in enumerate(entries, start=1):
        operation = entry["operation"] if entry["operation"] is not None else "text"
        lines.append(f"{index}. {_code(_datetime(entry['date']))} — {_code(operation)}")
        lines.append(f"   - Parameters: {_value(entry['parameters'])}")
        lines.append(f"   - Message: {_value(entry['message'])}")
    return lines


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
    return lines


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
    if any(character in target for character in " \t&()<>\"'`"):
        raise ValueError(
            "figure target must not contain whitespace or Markdown control characters"
        )
    if not target.isprintable():
        raise ValueError("figure target must not contain control characters")
    if not _has_safe_percent_encoding(target):
        raise ValueError(
            "figure target must use valid percent-encoding without control characters"
        )
    decoded_target = unquote(target)
    if not decoded_target.isprintable() or "\\" in decoded_target:
        raise ValueError(
            "figure target must be a safe relative POSIX path without control "
            "characters or backslashes"
        )
    try:
        parsed = urlparse(target)
    except ValueError as exc:
        raise ValueError(
            "figure target must be a valid URL or safe relative POSIX path"
        ) from exc
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        decoded_path = PurePosixPath(unquote(parsed.path))
        if "." in decoded_path.parts or ".." in decoded_path.parts:
            raise ValueError("figure target must not contain dot path segments")
        return target
    path = PurePosixPath(decoded_target)
    if (
        parsed.scheme
        or path.is_absolute()
        or decoded_target.startswith("~")
        or "." in path.parts
        or ".." in path.parts
    ):
        raise ValueError(
            "figure target must be an HTTP(S) URL or a safe relative POSIX path"
        )
    return target


def _has_safe_percent_encoding(value: str) -> bool:
    index = value.find("%")
    while index != -1:
        escape = value[index + 1 : index + 3]
        if len(escape) != 2 or any(
            character not in _HEX_DIGITS for character in escape
        ):
            return False
        byte = int(escape, 16)
        if byte < 0x20 or byte == 0x7F or byte in {0x2E, 0x2F, 0x5C}:
            return False
        index = value.find("%", index + 3)
    return True


def _datetime(value: datetime) -> str:
    return value.isoformat()


def _unit(value) -> str:
    return "—" if value is None else _code(str(value))


def _value(value, *, depth: int = 0, _budget=None) -> str:
    budget = (
        _budget if _budget is not None else [_MAX_RENDERED_NODES, _MAX_RENDERED_CHARS]
    )
    if budget[0] <= 0 or budget[1] <= 0:
        return _RENDER_BUDGET
    budget[0] -= 1
    if value is None:
        return "—"
    if isinstance(value, Path):
        return _REDACTED_PATH
    if isinstance(value, str):
        rendered = _text(value)
        if len(rendered) > budget[1]:
            return _RENDER_BUDGET
        budget[1] -= len(rendered)
        return rendered
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral | np.integer):
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
        magnitude = _value(value.magnitude, depth=depth + 1, _budget=budget)
        return f"{magnitude} {_unit(value.units)}"
    if isinstance(value, Mapping):
        if depth >= _MAX_NESTING_DEPTH:
            return "[omitted: maximum depth]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omitted: collection too large]"
        pairs = []
        for key in sorted(value, key=str):
            if budget[0] <= 0 or budget[1] <= 0:
                pairs.append(_RENDER_BUDGET)
                break
            rendered = _value(value[key], depth=depth + 1, _budget=budget)
            pairs.append(f"{_text(str(key))}: {rendered}")
        return "{" + ", ".join(pairs) + "}"
    if isinstance(value, list | tuple):
        if depth >= _MAX_NESTING_DEPTH:
            return "[omitted: maximum depth]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omitted: collection too large]"
        items = []
        for item in value:
            if budget[0] <= 0 or budget[1] <= 0:
                items.append(_RENDER_BUDGET)
                break
            items.append(_value(item, depth=depth + 1, _budget=budget))
        return "[" + ", ".join(items) + "]"
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
        return "[omitted: text too long]"
    return _escape(value)


def _escape(value: str) -> str:
    value = value.replace("\\", "\\\\")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", "\\n")
    for character in "`*_{}[]<>#+-.!|()~":
        value = value.replace(character, f"\\{character}")
    return value


def _code(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    delimiter = "`"
    while delimiter in value:
        delimiter += "`"
    if not value or value.startswith("`") or value.endswith("`"):
        value = f" {value} "
    return f"{delimiter}{value}{delimiter}"
