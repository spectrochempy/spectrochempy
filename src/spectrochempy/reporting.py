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
from urllib.parse import urlparse

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import Unit

__all__ = ["FigureReference", "render_dataset_markdown"]


_SCOPE_MESSAGE = (
    "Ce rapport synthétise les objets et l’historique fournis. Il ne constitue pas "
    "une recette exécutable ni une preuve de provenance exhaustive."
)
_MAX_COLLECTION_ITEMS = 32
_MAX_NESTING_DEPTH = 4
_MAX_TEXT_LENGTH = 4096
_MAX_RENDERED_NODES = 256
_MAX_RENDERED_CHARS = 16384
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SUFFIX = re.compile(r"\.([A-Za-z][A-Za-z0-9]*|[0-9]+)$")
_NUMERIC_STEM = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+)*$")
_MASKED_PATH = "[masqué : chemin ou nom de fichier]"
_RENDER_BUDGET = "[omis : budget de rendu épuisé]"


@dataclass(frozen=True)
class FigureReference:
    """
    An explicit, report-relative or HTTPS figure reference.

    Parameters
    ----------
    label
        Figure label, used for the section heading and the image alternative
        text.
    target
        HTTP(S) URL or safe relative POSIX path. The target is not checked for
        existence and is never copied.
    caption
        Optional caption rendered after the image.
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

    Parameters
    ----------
    dataset : NDDataset
        Dataset to describe. The renderer does not modify it.
    title : str
        Required document title.
    metadata : sequence of str, optional
        Explicit metadata keys to include, in report order.
    figures : sequence of FigureReference, optional
        Explicit figure references. Matplotlib figures and implicit capture are
        not supported.

    Returns
    -------
    str
        Deterministic Markdown document ending with a single newline.
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
        f"# {_escape(title)}",
        "",
        "## Portée et limites",
        "",
        f"> {_SCOPE_MESSAGE}",
    ]
    lines.extend(["", "## Jeux de données", "", f"### {_dataset_heading(dataset)}", ""])
    lines.extend(_dataset_summary(dataset))
    lines.extend(_coordinates(dataset))
    lines.extend(_metadata(dataset, metadata))
    lines.extend(_history(dataset))
    lines.extend(_figures(figures))
    return "\n".join(lines).rstrip() + "\n"


def _dataset_heading(dataset: NDDataset) -> str:
    if dataset.has_defined_name:
        return _text(dataset.name)
    return "Jeu de données"


def _dataset_summary(dataset: NDDataset) -> list[str]:
    lines = ["- Forme: " + _code(repr(dataset.shape))]
    lines.append("- Dimensions: " + _code(", ".join(dataset.dims)))
    lines.append("- Unité des données: " + _unit(dataset.units))
    if dataset.title != "<untitled>":
        lines.append("- Titre: " + _text(dataset.title))
    origin = dataset.origin
    if isinstance(origin, str) and not origin.strip():
        origin = None
    lines.append("- Origine: " + _value(origin))
    return lines


def _coordinates(dataset: NDDataset) -> list[str]:
    lines = ["", "#### Coordonnées", ""]
    coordset = dataset.coordset
    if coordset is None:
        return [*lines, "- Aucune coordonnée fournie."]

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
            f"taille: {coord.size}",
            f"unité: {_unit(coord.units)}",
        ]
        extent = _coordinate_extent(coord)
        if extent is not None:
            summary.append(f"étendue: {extent}")
        lines.append("- " + "; ".join(summary))
    return lines


def _coordinate_extent(coord) -> str | None:
    data = coord.data
    if data is None or len(data) == 0:
        return None
    return f"{_value(data[0])} à {_value(data[-1])}"


def _metadata(dataset: NDDataset, keys: Sequence[str]) -> list[str]:
    if not keys:
        return []
    lines = ["", "#### Metadata sélectionnée", ""]
    for key in keys:
        lines.append(f"- {_code(key)}: {_value(dataset.meta.get(key))}")
    return lines


def _history(dataset: NDDataset) -> list[str]:
    lines = ["", "#### Historique", ""]
    entries = dataset.history_entries
    if not entries:
        return [*lines, "- Aucune entrée d’historique fournie."]

    for index, entry in enumerate(entries, start=1):
        operation = entry["operation"] if entry["operation"] is not None else "texte"
        lines.append(f"{index}. {_code(_datetime(entry['date']))} — {_code(operation)}")
        lines.append(f"   - Paramètres: {_value(entry['parameters'])}")
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
            lines.extend(["", _text(figure.caption)])
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
    if any(character in target for character in "\r\n\t ()<>"):
        raise ValueError(
            "figure target must not contain whitespace or Markdown control characters"
        )
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return target
    path = Path(target)
    if (
        parsed.scheme
        or path.is_absolute()
        or target.startswith("~")
        or "\\" in target
        or ".." in path.parts
    ):
        raise ValueError(
            "figure target must be an HTTP(S) URL or a safe relative POSIX path"
        )
    return target


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
        return _MASKED_PATH
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
            return "[omis : profondeur maximale]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omis : collection trop grande]"
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
            return "[omis : profondeur maximale]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omis : collection trop grande]"
        items = []
        for item in value:
            if budget[0] <= 0 or budget[1] <= 0:
                items.append(_RENDER_BUDGET)
                break
            items.append(_value(item, depth=depth + 1, _budget=budget))
        return "[" + ", ".join(items) + "]"
    return f"[omis : {type(value).__name__}]"


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


def _looks_like_path(value: str) -> bool:
    """
    Return whether free text is more likely a path, file name or host.

    Structural prefixes are conclusive: ``/…``, ``~/``, ``./``, ``../``,
    Windows drives and backslashes. A trailing extension is also conclusive
    whatever the whitespace: a letter-initial extension masks
    ``private report.csv`` or ``report.spectrochempy`` whatever its length,
    while a numeric one masks a word-stem token such as ``report.123456789``.

    A bare ``/`` is not conclusive, so scientific text such as ``N/A``,
    ``mg/L``, ``and/or``, ``24/7``, ``absorbance/a.u.`` or a DOI stays
    rendered; consequently an extension-less relative path without ``./`` or
    ``../`` is not masked. Decimals and versions (``1.5``, ``1.5.3``,
    ``version 1.12``) stay text, and so do extension-less file names
    (``1r``, ``acqus``), which would need an explicit allowlist.
    """
    if value.startswith(("/", "~/", "./", "../")) or _WINDOWS_PATH.match(value):
        return True
    if "\\" in value:
        return True
    match = _SUFFIX.search(value)
    if match is None:
        return False
    extension = match.group(1)
    if extension[:1].isalpha():
        return True
    if any(character.isspace() for character in value):
        return False
    if "/" in value:
        return False
    if len(extension) == 1:
        return False
    return not _NUMERIC_STEM.match(value[: match.start()])


def _text(value: str) -> str:
    if len(value) > _MAX_TEXT_LENGTH:
        return "[omis : texte trop long]"
    if _looks_like_path(value):
        return _MASKED_PATH
    return _escape(value)


def _escape(value: str) -> str:
    value = value.replace("\\", "\\\\")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", "\\n")
    for character in "`*_{}[]<>#+-.!|()":
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
