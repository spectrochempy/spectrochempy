"""
Utility for examples that require optional SpectroChemPy plugins.

Usage in a sphinx-gallery example::

    from _plugin_deps import requires_plugin

    if not requires_plugin("spectrochempy_iris", "spectrochempy-iris"):
        # plugin not available — early return or skip
    else:
        # … actual example code …
"""

from __future__ import annotations

from importlib.util import find_spec


def requires_plugin(package_name: str, dist_name: str | None = None) -> bool:
    """
    Check whether an optional plugin package is installed.

    When the plugin is **not** found, a human-readable message is printed so
    that it appears as the first cell output of the rendered gallery example.

    Parameters
    ----------
    package_name : str
        The Python import name of the plugin, e.g. ``"spectrochempy_iris"``.
    dist_name : str, optional
        The PyPI / conda distribution name, e.g. ``"spectrochempy-iris"``.
        When ``None`` the *package_name* is used after replacing underscores
        with dashes.

    Returns
    -------
    bool
        ``True`` when the plugin can be imported, ``False`` otherwise.
    """
    if find_spec(package_name) is not None:
        return True

    if dist_name is None:
        dist_name = package_name.replace("_", "-")

    print(f"[plugin required] {dist_name}")
    print("-" * (len(dist_name) + 20))
    print(f"This example requires the optional '{dist_name}' plugin.")
    print(f"Install with: pip install {dist_name}")
    return False
