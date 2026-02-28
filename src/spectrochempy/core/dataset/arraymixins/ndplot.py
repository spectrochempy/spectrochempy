"""
INTERNAL MODULE REMOVED.

The NDPlot mixin has been permanently removed as part of the
plotting decoupling (Phase 3-C).

Plotting is now handled by:

- dataset.plot()
- spectrochempy.plotting.plot1d
- spectrochempy.plotting.plot2d
- spectrochempy.plotting.plot3d

NDPlot was an internal implementation detail and is not part of the
public API.

If you relied on this module directly, migrate to the new plotting API.
"""

# No matplotlib imports - this module is now removed.


def __getattr__(name):
    raise ImportError(
        "spectrochempy.core.dataset.arraymixins.ndplot has been removed. "
        "Use dataset.plot() or functions from spectrochempy.plotting instead."
    )
