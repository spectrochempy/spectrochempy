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
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FILE_EXTENSION = re.compile(r"^\.[A-Za-z][A-Za-z0-9]{0,7}$")
_MASKED_PATH = "[masqué : chemin ou nom de fichier]"


@dataclass(frozen=True)
class FigureReference:
    """An explicit, report-relative or HTTPS figure reference."""

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
    dataset
        Dataset to describe. The renderer does not modify it.
    title
        Required document title.
    metadata
        Explicit metadata keys to include, in report order.
    figures
        Explicit figure references. Matplotlib figures and implicit capture are
        not supported.
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
        return _escape(dataset.name)
    return "Jeu de données"


def _dataset_summary(dataset: NDDataset) -> list[str]:
    lines = ["- Forme: " + _code(repr(dataset.shape))]
    lines.append("- Dimensions: " + _code(", ".join(dataset.dims)))
    lines.append("- Unité des données: " + _unit(dataset.units))
    if dataset.title != "<untitled>":
        lines.append("- Titre: " + _escape(dataset.title))
    if dataset.origin is not None:
        lines.append("- Origine: " + _safe_text(dataset.origin))
    return lines


def _coordinates(dataset: NDDataset) -> list[str]:
    lines = ["", "#### Coordonnées", ""]
    if dataset.coordset is None:
        return [*lines, "- Aucune coordonnée fournie."]

    for dim in dataset.dims:
        coord = dataset.coord(dim)
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
            f"{_code(dim)} — {_escape(label)}",
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
        lines.append(f"   - Message: {_escape(entry['message'])}")
    return lines


def _figures(figures: Sequence[FigureReference]) -> list[str]:
    if not figures:
        return []
    lines = ["", "## Figures", ""]
    for figure in figures:
        target = _figure_target(figure.target)
        lines.append(f"### {_escape(figure.label)}")
        lines.extend(["", f"![{_escape(figure.label)}]({target})"])
        if figure.caption is not None:
            lines.extend(["", _escape(figure.caption)])
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
    if any(character in target for character in "\r\n()<>"):
        raise ValueError("figure target must not contain Markdown control characters")
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


def _value(value, *, depth: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, Path):
        return _MASKED_PATH
    if isinstance(value, str):
        if len(value) > _MAX_TEXT_LENGTH:
            return "[omis : texte trop long]"
        if _looks_like_path(value):
            return _MASKED_PATH
        return _escape(value)
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
        return f"{_value(value.magnitude, depth=depth + 1)} {_unit(value.units)}"
    if isinstance(value, Mapping):
        if depth >= _MAX_NESTING_DEPTH:
            return "[omis : profondeur maximale]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omis : collection trop grande]"
        pairs = [
            f"{_escape(str(key))}: {_value(value[key], depth=depth + 1)}"
            for key in sorted(value, key=str)
        ]
        return "{" + ", ".join(pairs) + "}"
    if isinstance(value, list | tuple):
        if depth >= _MAX_NESTING_DEPTH:
            return "[omis : profondeur maximale]"
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[omis : collection trop grande]"
        return "[" + ", ".join(_value(item, depth=depth + 1) for item in value) + "]"
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

    Structural separators are conclusive. A suffix-like dotted token only counts
    when the value has no whitespace and the extension is file-like
    (``report.csv``, ``example.com``), so sentences, versions and decimals such as
    ``TopSpin 4.0``, ``600.13 MHz`` or ``1.5`` stay scientific text.

    Extension-less file names (``1r``, ``acqus``) are not masked; recognizing them
    would require an explicit allowlist rather than a shape heuristic.
    """
    if value.startswith(("/", "~/", "./", "../")) or _WINDOWS_PATH.match(value):
        return True
    if "\\" in value:
        return True
    if any(character.isspace() for character in value):
        return False
    if "/" in value:
        return True
    return bool(_FILE_EXTENSION.match(Path(value).suffix))


def _safe_text(value) -> str:
    return _value(value)


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
    return f"{delimiter}{value}{delimiter}"
