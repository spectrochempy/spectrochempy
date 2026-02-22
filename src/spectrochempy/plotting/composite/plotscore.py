# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Score plot composite function.

This module provides a standalone plot_score function for visualizing
PCA scores in 2D or 3D scatter plots.
"""

__all__ = ["plot_score"]

import numpy as np
from matplotlib.lines import Line2D

from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.mplutils import show as mpl_show


def plot_score(
    scores,
    components=(1, 2),
    *,
    ax=None,
    clear=True,
    cmap=None,
    color=None,
    color_mapping="index",
    show_labels=False,
    labels_column=None,
    elev=None,
    azim=None,
    show=True,
):
    """
    Plot PCA scores as a 2D or 3D scatter plot.

    Parameters
    ----------
    scores : array-like or NDDataset
        Scores data with shape (n_samples, n_components).
        If an NDDataset, the data attribute is used.
    components : tuple of int, optional
        Principal components to plot (1-based indexing).
        Length 2 for 2D plot, length 3 for 3D plot.
        Default: (1, 2).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
        Only used when ``ax`` is provided.
    cmap : str or Colormap, optional
        Colormap for coloring points. Default: "viridis".
    color : color or array-like, optional
        If provided, use this color for all points (single color)
        or as color values for each point. Overrides color_mapping.
    color_mapping : {"index", "labels"}, optional
        Method for mapping colors to points:

        - ``"index"`` (default): Sequential colors by sample index.
        - ``"labels"``: Color by categorical labels from scores.y.labels.
          Adds a legend showing label categories.

    show_labels : bool, optional
        If True, annotate each point with its label from scores.y.labels.
        Default: False.
    labels_column : int, optional
        Column index in scores.y.labels to use (0-based).
        If None, uses column 0 for color_mapping, or last column for show_labels.
    elev : float, optional
        Elevation angle (degrees) for 3D plots. If None, uses preferences.axes3d_elev.
    azim : float, optional
        Azimuth angle (degrees) for 3D plots. If None, uses preferences.axes3d_azim.
    show : bool, optional
        Whether to display the figure. Default: True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the scatter plot.

    Raises
    ------
    ValueError
        If components has invalid length (not 2 or 3).
        If component index exceeds available components.
        If color_mapping="labels" but scores has no labels.
        If show_labels=True but scores has no y coordinate or labels.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrochempy import plot_score
    >>> scores = np.random.randn(50, 5)
    >>> ax = plot_score(scores, components=(1, 2), show=False)

    With label-based coloring:

    >>> ax = plot_score(scores, color_mapping="labels", show=False)  # doctest: +SKIP
    """
    if color_mapping not in ("index", "labels"):
        raise ValueError("color_mapping must be 'index' or 'labels'")

    from spectrochempy.application.preferences import preferences as prefs

    if cmap is None:
        cmap = prefs.colormap

    if hasattr(scores, "masked_data"):
        data = scores.masked_data
    elif hasattr(scores, "data"):
        data = np.asarray(scores.data)
    else:
        data = np.asarray(scores)

    n_samples, n_components_available = data.shape

    if len(components) not in (2, 3):
        raise ValueError(f"components must have length 2 or 3, got {len(components)}")

    components = tuple(components)

    for pc in components:
        if pc < 1 or pc > n_components_available:
            raise ValueError(
                f"component {pc} is out of range [1, {n_components_available}]"
            )

    pcs = [pc - 1 for pc in components]
    n_dims = len(pcs)

    legend_needed = False
    unique_categories = None

    if color is not None:
        color_values = color
    elif color_mapping == "index":
        color_values = np.arange(n_samples)
    else:
        if not hasattr(scores, "y") or scores.y is None:
            raise ValueError("No labels available for color_mapping='labels'.")

        if getattr(scores.y, "labels", None) is None:
            raise ValueError("No labels available for color_mapping='labels'.")

        labels_array = np.asarray(scores.y.labels)

        if labels_array.ndim == 1:
            col_idx = 0
            if labels_column is not None and labels_column != 0:
                raise ValueError(
                    f"labels_column {labels_column} out of range for 1D labels."
                )
            selected_labels = labels_array
        else:
            n_cols = labels_array.shape[1]

            if labels_column is None:
                col_idx = 0
            else:
                if not isinstance(labels_column, int):
                    raise ValueError("labels_column must be an integer.")
                if labels_column < 0 or labels_column >= n_cols:
                    raise ValueError(
                        f"labels_column {labels_column} out of range [0, {n_cols - 1}]."
                    )
                col_idx = labels_column

            selected_labels = labels_array[:, col_idx]

        unique_categories = sorted(str(lab) for lab in np.unique(selected_labels))
        label_to_index = {lab: i for i, lab in enumerate(unique_categories)}
        color_values = np.array([label_to_index[str(lab)] for lab in selected_labels])
        legend_needed = True

    labels = None
    if show_labels:
        if not hasattr(scores, "y") or scores.y is None:
            raise ValueError("Scores dataset has no y coordinate for labeling.")

        if getattr(scores.y, "labels", None) is None:
            raise ValueError("Scores.y.labels is empty.")

        labels_array = scores.y.labels
        if labels_array.ndim == 1:
            labels = [str(lab) for lab in labels_array]
        else:
            if labels_column is None:
                col_idx = labels_array.shape[1] - 1
            else:
                if labels_column < 0 or labels_column >= labels_array.shape[1]:
                    raise ValueError(
                        f"labels_column {labels_column} is out of range "
                        f"[0, {labels_array.shape[1] - 1}]"
                    )
                col_idx = labels_column
            labels = [str(lab) for lab in labels_array[:, col_idx]]

        if len(labels) != n_samples:
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match "
                f"number of samples ({n_samples})"
            )

    if n_dims == 2:
        if ax is None:
            fig = get_figure()
            ax = fig.add_subplot(111)
        elif clear:
            ax.clear()

        x = data[:, pcs[0]]
        y = data[:, pcs[1]]

        scatter = ax.scatter(x, y, c=color_values, cmap=cmap)
        ax.set_xlabel(f"PC{components[0]}")
        ax.set_ylabel(f"PC{components[1]}")

        if legend_needed and unique_categories is not None:
            handles = []
            for i, cat in enumerate(unique_categories):
                color_rgba = scatter.cmap(scatter.norm(i))

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=color_rgba,
                        label=cat,
                    )
                )
            ax.legend(handles=handles, loc="best")

        if labels is not None:
            for i in range(n_samples):
                ax.text(x[i], y[i], labels[i], fontsize=8, ha="left", va="bottom")

    elif n_dims == 3:
        if ax is None:
            fig = get_figure()
            ax = fig.add_subplot(111, projection="3d")
        elif clear:
            ax.clear()

        x = data[:, pcs[0]]
        y = data[:, pcs[1]]
        z = data[:, pcs[2]]

        scatter = ax.scatter(x, y, z, c=color_values, cmap=cmap)
        ax.set_xlabel(f"PC{components[0]}")
        ax.set_ylabel(f"PC{components[1]}")
        ax.set_zlabel(f"PC{components[2]}")

        from spectrochempy.application.preferences import preferences as prefs

        _elev = elev if elev is not None else prefs.axes3d_elev
        _azim = azim if azim is not None else prefs.axes3d_azim
        ax.view_init(elev=_elev, azim=_azim)

        if legend_needed and unique_categories is not None:
            handles = []
            for i, cat in enumerate(unique_categories):
                color_rgba = scatter.cmap(scatter.norm(i))

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=color_rgba,
                        label=cat,
                    )
                )
            ax.legend(handles=handles, loc="best")

        if labels is not None:
            for i in range(n_samples):
                ax.text(x[i], y[i], z[i], labels[i], fontsize=8)

    if show:
        mpl_show()

    return ax
