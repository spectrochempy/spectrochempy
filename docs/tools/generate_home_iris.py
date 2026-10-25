"""
Generate the homepage's scientific illustrations from the IRIS gallery data.

Run manually with the optional spectrochempy-iris plugin installed:
    python docs/tools/generate_home_iris.py

Scientific source:
plugins/spectrochempy-iris/examples/analysis/a_decomposition/plot_iris_intro.py
The contours use its automatic regularization search and selected result (5).
Only the saved SVGs are needed to build the documentation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

import spectrochempy as scp


def main():
    """Render measured spectra and the recovered adsorption distribution."""
    output = Path(__file__).resolve().parents[1] / "_static" / "images"
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "scp-home-iris"
    pressures = [
        0.003,
        0.004,
        0.009,
        0.014,
        0.021,
        0.026,
        0.036,
        0.051,
        0.093,
        0.150,
        0.203,
        0.300,
        0.404,
        0.503,
        0.602,
        0.702,
        0.801,
        0.905,
        1.004,
    ]
    data = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
    data.y = [data.y.copy(), scp.Coord(pressures, title="pressure", units="torr")]
    data.y.select(2)
    data = data[:, 2250.0:1950.0]
    kernel = data.iris.kernel_matrix(kernel_type="langmuir", q=[-8, -1, 50])
    model = scp.iris.IRIS(log_level="INFO", reg_par=[-6, -2])
    model.fit(data, kernel)
    distribution = model.result.f[5].squeeze()

    def canvas():
        fig = plt.figure(figsize=(2.8, 3.3))
        ax = fig.add_axes((0.02, 0.03, 0.96, 0.94))
        ax.set_axis_off()
        return fig, ax

    def save(fig, name, bounds=None):
        fig.savefig(
            output / f"home-iris-{name}.svg",
            transparent=True,
            bbox_inches=bounds,
            metadata={"Date": None, "Creator": "SpectroChemPy / Matplotlib"},
        )
        plt.close(fig)

    fig, ax = canvas()
    colors = ["#47e4e0", "#57cfe9", "#6aafff", "#a59af8", "#d4a1f0", "#f3bee8"]
    for index, color in zip([0, 3, 6, 10, 14, 18], colors, strict=True):
        ax.plot(data.x.data, data.data[index], color=color, linewidth=1.4)
    ax.set_xlim(2250, 1950)
    ax.set_ylim(-0.008, 0.27)
    save(fig, "spectra")

    fig, ax = canvas()
    levels = np.geomspace(
        distribution.data.max() * 0.025, distribution.data.max() * 0.95, 9
    )
    palette = [
        "#258dc5",
        "#22b6d4",
        "#33cfd2",
        "#6bdcc8",
        "#a6e5b3",
        "#d8e99b",
        "#f5e588",
        "#ffd571",
        "#ffe9a0",
    ]
    filled = ax.contourf(
        distribution.x.data,
        distribution.y.data,
        distribution.data,
        levels=levels,
        colors=palette,
        alpha=0.4,
        extend="max",
    )
    contours = ax.contour(
        distribution.x.data,
        distribution.y.data,
        distribution.data,
        levels=levels,
        colors=palette,
        linewidths=1.05,
    )
    ax.set_xlim(2250, 1950)
    ax.set_ylim(-8, -1)
    bounds = Bbox.union(
        [
            path.get_extents(ax.transData)
            for artist in (filled, contours)
            for path in artist.get_paths()
            if path.vertices.size
        ]
    )
    bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
    save(fig, "insights", bounds)
    scp.info_(f"Saved illustrations to {output}; lambda = {model.lambdas.data[5]}")


if __name__ == "__main__":
    main()
