# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Preparation and rendering of coordinate-aware marginal panels."""

from collections.abc import Mapping
from numbers import Real

import numpy as np
from matplotlib.colors import is_color_like
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.utils.mplutils import make_label


def _coord(dataset, dim):
    coord = dataset.coord(dim)
    return getattr(coord, "default", coord)


def _positions(coord, size):
    if coord is None or coord.is_empty or coord.show_datapoints:
        return np.arange(size), None
    values = coord.data
    if coord.is_labeled and not np.any(values):
        return np.arange(1, size + 1), None
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("Projection coordinates must be finite real numbers")
    return values, coord.units


def _convert(values, units, target):
    if units == target:
        return values
    if units is None or target is None:
        raise ValueError("Projection references must have compatible units")
    try:
        return Quantity(values, units).to(target).magnitude
    except (TypeError, ValueError) as exc:
        raise ValueError("Projection references must have compatible units") from exc


def _expand(value, count, name):
    if name == "color" and (value is None or is_color_like(value)):
        return [value] * count
    if isinstance(value, (str, int, float)) or value is None:
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(f"Projection {name} must have one value per trace")
    return values


def _prepare_projections(dataset, method, kwargs, prefs):
    """Resolve retained dimensions and reference curves without changing inputs."""
    projection_options = {
        "projections",
        "show_projections",
        "show_projection_x",
        "show_projection_y",
        "projection_data",
    }
    if method not in (
        "contour",
        "contourf",
        "map",
        "image",
    ) and not projection_options.intersection(kwargs):
        return {}
    data = kwargs.get("projection_data", {})
    styles = kwargs.get("projection_kwargs", {})
    if not isinstance(data, Mapping) or not isinstance(styles, Mapping):
        raise TypeError("projection_data and projection_kwargs must be mappings")
    dimensions = tuple(dataset.dims)
    selection = kwargs.get("projections")
    if selection is None:
        all_axes = kwargs.get("show_projections", prefs.show_projections)
        selection = [
            dim
            for dim in dimensions
            if all_axes
            or kwargs.get(
                f"show_projection_{dim}",
                getattr(prefs, f"show_projection_{dim}", False),
            )
        ]
        selection.extend(dim for dim in data if dim not in selection)
    elif isinstance(selection, (bool, np.bool_)):
        selection = list(dimensions) if selection else []
    elif isinstance(selection, str):
        selection = [selection]
    else:
        try:
            selection = list(selection)
        except TypeError as exc:
            raise TypeError("projections must be a bool or dimension names") from exc
    if any(not isinstance(dim, str) or dim not in dimensions for dim in selection):
        raise ValueError(f"Projection dimensions must belong to {dimensions}")
    if len(set(selection)) != len(selection):
        raise ValueError("Projection dimensions must be unique")
    if set(data) - set(selection) or set(styles) - set(selection):
        raise ValueError("Projection data and styles require a selected dimension")
    if not selection:
        return {}
    if method not in ("contour", "contourf", "map", "image"):
        raise ValueError("Projections require a contour or contourf plot")
    size = kwargs.get("projection_size", 0.25)
    pad = kwargs.get("projection_pad", 0.12)
    if (
        not isinstance(size, Real)
        or isinstance(size, bool)
        or not np.isfinite(size)
        or size <= 0
    ):
        raise ValueError("projection_size must be a finite positive ratio")
    if (
        not isinstance(pad, Real)
        or isinstance(pad, bool)
        or not np.isfinite(pad)
        or pad < 0
    ):
        raise ValueError("projection_pad must be finite and nonnegative")
    reducer = kwargs.get("projection_reducer", "sum")
    reducers = {"sum": np.ma.sum, "mean": np.ma.mean, "max": np.ma.max}
    if not isinstance(reducer, str) or reducer not in reducers:
        raise ValueError("projection_reducer must be 'sum', 'mean', or 'max'")
    values = np.ma.masked_invalid(
        (dataset.imag if kwargs.get("imag", False) else dataset.real).masked_data
    )
    prepared = {}
    for dim in selection:
        axis = dimensions.index(dim)
        positions, units = _positions(_coord(dataset, dim), dataset.shape[axis])
        if dim not in data:
            traces = [(positions, reducers[reducer](values, axis=1 - axis))]
            intensity_label = make_label(dataset, "values")
        else:
            refs = data[dim]
            refs = [refs] if isinstance(refs, NDDataset) else list(refs)
            if not refs or any(not isinstance(ref, NDDataset) for ref in refs):
                raise TypeError("Projection references must be NDDataset objects")
            traces = []
            signal_units = refs[0].units
            intensity_label = make_label(refs[0], "values")
            for ref in refs:
                if ref.ndim not in (1, 2):
                    raise ValueError("Projection references must be 1D or 2D")
                ref_positions, ref_units = _positions(
                    _coord(ref, ref.dims[-1]), ref.shape[-1]
                )
                ref_positions = _convert(ref_positions, ref_units, units)
                if not np.isfinite(ref_positions).all():
                    raise ValueError("Projection coordinates must be finite")
                if max(ref_positions) < min(positions) or min(ref_positions) > max(
                    positions
                ):
                    raise ValueError(
                        "Projection reference coordinates do not overlap the map"
                    )
                if ref.units == signal_units:
                    ref_values = ref.real.masked_data
                else:
                    if ref.units is None or signal_units is None:
                        raise ValueError(
                            "Projection references must have compatible units"
                        )
                    try:
                        ref_values = ref.real.to(signal_units).masked_data
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Projection references must have compatible units"
                        ) from exc
                for row in np.ma.asarray(ref_values).reshape(-1, ref.shape[-1]):
                    traces.append((ref_positions, row))
        options = styles.get(dim, {})
        if not isinstance(options, Mapping):
            raise TypeError("Projection style for each dimension must be a mapping")
        allowed = {
            "color",
            "linestyle",
            "linewidth",
            "labels",
            "legend",
            "intensity_label",
            "title",
        }
        if set(options) - allowed:
            raise ValueError(
                f"Unsupported projection options: {set(options) - allowed}"
            )
        count = len(traces)
        default_labels = (
            [f"Reference {i + 1}" for i in range(count)]
            if dim in data
            else [f"{reducer} over {dimensions[1 - axis]}"]
        )
        labels = options.get("labels", default_labels)
        if labels is not None and (isinstance(labels, str) or len(labels) != count):
            raise ValueError("Projection labels must have one value per trace")
        prepared["top" if axis == 1 else "right"] = {
            "traces": traces,
            "colors": _expand(options.get("color"), count, "color"),
            "linestyles": _expand(options.get("linestyle", "-"), count, "linestyle"),
            "linewidths": _expand(options.get("linewidth", 1.0), count, "linewidth"),
            "labels": labels,
            "legend": options.get("legend", False),
            "title": options.get("title", ""),
            "label": options.get("intensity_label", intensity_label),
            "size": float(size),
            "pad": float(pad),
        }
    return prepared


def _draw_projections(ax, prepared, *, clear):
    """Attach aligned marginal axes to a main axes, including explicit subplots."""
    previous = getattr(ax, "_scp_projection_axes", {})
    if previous and not clear and set(previous) != set(prepared):
        if prepared:
            raise ValueError("clear=False cannot change the projection panels")
        return
    if clear:
        for marginal in previous.values():
            marginal.remove()
        if previous:
            ax.set_axes_locator(None)
        previous = {}
    if not prepared:
        ax._scp_projection_axes = previous
        return
    divider = None if previous else make_axes_locatable(ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    marginals = {}
    for side, spec in prepared.items():
        marginal = previous.get(side)
        if marginal is None:
            shared = {"sharex": ax} if side == "top" else {"sharey": ax}
            marginal = divider.append_axes(
                side, size=f"{100 * spec['size']}%", pad=spec["pad"], **shared
            )
        for i, (positions, values) in enumerate(spec["traces"]):
            xy = (positions, values) if side == "top" else (values, positions)
            marginal.plot(
                *xy,
                color=spec["colors"][i],
                linestyle=spec["linestyles"][i],
                linewidth=spec["linewidths"][i],
                label=None if spec["labels"] is None else spec["labels"][i],
            )
        if side == "top":
            marginal.tick_params(labelbottom=False)
            marginal.set_ylabel(spec["label"])
        else:
            marginal.tick_params(labelleft=False)
            marginal.set_xlabel(spec["label"])
        marginal.set_title(spec["title"])
        marginal._scp_projection_title = spec["title"]
        if spec["legend"]:
            marginal.legend(
                loc=spec["legend"] if isinstance(spec["legend"], str) else "best"
            )
        marginals[side] = marginal
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax._scp_projection_axes = marginals
    if "top" in marginals:
        _set_panel_title(ax, ax.get_title())


def _set_panel_title(ax, title, **kwargs):
    top = getattr(ax, "_scp_projection_axes", {}).get("top")
    if top is None:
        ax.set_title(title, **kwargs)
        return
    ax.set_title("")
    projection_title = getattr(top, "_scp_projection_title", "")
    top.set_title(
        "\n".join(text for text in (title, projection_title) if text),
        **(kwargs if title else {}),
    )


def _panel_axes(ax):
    """Return the public roles of the axes belonging to one 2D panel."""
    return {"main": ax, **getattr(ax, "_scp_projection_axes", {})}
