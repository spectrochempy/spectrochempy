# ======================================================================================
# Copyright (©) 2015-2026 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# Authors:
# christian.fernandez@ensicaen.fr
# arnaud.travert@ensicaen.fr
#
# This software is a computer program whose purpose is to provide a framework
# for processing, analysing and modelling *Spectro*scopic
# data for *Chem*istry with *Py*thon (SpectroChemPy). It is is a cross
# platform software, running on Linux, Windows or OS X.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================================

# ======================================================================================
"""
SpectroChemPy API Module.

SpectroChemPy is a framework for processing, analyzing and modeling Spectroscopic data
for Chemistry with Python. It is a cross-platform software, running on Linux, Windows or OS X.

This module serves as the main entry point for the SpectroChemPy package, providing:
- Configuration of warning filters
- Import of the public API
- Dynamic attribute access to NDDataset methods
"""

import warnings

import lazy_loader as _lazy_loader

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
# Store the original __getattr__ and __dir__ from lazy_loader
original_getattr, original_dir, *_ = _lazy_loader.attach_stub(__name__, __file__)

# Dictionary mapping top-level objects to their module paths

from spectrochempy.lazyimport.api_methods import _LAZY_IMPORTS
from spectrochempy.lazyimport.dataset_methods import _LAZY_DATASETS_IMPORTS

from . import application

# --------------------------------------------------------------------------------------
# Warning configurations
# --------------------------------------------------------------------------------------
application.start.set_warnings()

# --------------------------------------------------------------------------------------
# Plugin  manager
# --------------------------------------------------------------------------------------
from spectrochempy.plugins.features import KNOWN_PLUGIN_NAMESPACES
from spectrochempy.plugins.features import plugin_namespace_install_hint
from spectrochempy.plugins.features import plugin_reader_missing_stub
from spectrochempy.plugins.manager import plugin_manager
from spectrochempy.plugins.registry import registry

_EMITTED_PLUGIN_ROOT_WARNINGS: set[str] = set()

# --------------------------------------------------------------------------------------
# Plot profile API (lazy loaded)
# --------------------------------------------------------------------------------------
# These are exposed at top-level for convenience
_PLOT_PROFILE_FUNCTIONS = {
    "set_plot_profile": "spectrochempy.plotting.profile",
    "get_plot_profile": "spectrochempy.plotting.profile",
    "list_plot_profiles": "spectrochempy.plotting.profile",
    "save_plot_profile": "spectrochempy.plotting.profile",
    "delete_plot_profile": "spectrochempy.plotting.profile",
}


def __dir__():
    names = set(original_dir()) if callable(original_dir) else set()
    names.update(_PLOT_PROFILE_FUNCTIONS)
    names.update(_namespace_names())
    # Include already-discovered plugin readers and extensions
    # without triggering entry-point scanning (dir() should be side-effect free).
    names.update(_reader_names())
    names.update(_extension_names())
    names.update(_root_export_names())
    return sorted(names)


def _reader_names():
    return {f"read_{name}" for name in registry.available_readers}


def _namespace_names():
    return set(KNOWN_PLUGIN_NAMESPACES)


def _extension_names():
    names = set()
    for category in ("analysis", "simulation"):
        for entry_name in registry.extensions.list_category(category):
            names.add(entry_name)
    return names


def _root_export_names():
    names = set()
    for plugin in registry.available_plugins.values():
        names.update(getattr(plugin, "root_exports", {}))
    return names


def _normalise_root_export(plugin, alias, export):
    if isinstance(export, str):
        export = {"target": export}
    if not isinstance(export, dict):
        return None
    target = export.get("target", alias)
    namespace = export.get("namespace", getattr(plugin, "name", None))
    if not target or not namespace:
        return None
    return {
        "deprecated": bool(export.get("deprecated", False)),
        "namespace": namespace,
        "replacement": export.get("replacement", f"scp.{namespace}.{target}"),
        "target": target,
    }


def _plugin_root_export(name, namespace_cls):
    for plugin in plugin_manager.list_plugins():
        root_exports = getattr(plugin, "root_exports", {})
        if name not in root_exports:
            continue
        export = _normalise_root_export(plugin, name, root_exports[name])
        if export is None:
            continue
        if export["deprecated"] and name not in _EMITTED_PLUGIN_ROOT_WARNINGS:
            warnings.warn(
                f"scp.{name} is deprecated since SpectroChemPy 0.9.0 "
                f"and will be removed in 0.10.0. "
                f"Use {export['replacement']} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            _EMITTED_PLUGIN_ROOT_WARNINGS.add(name)
        namespace = namespace_cls(export["namespace"], plugin_manager, registry)
        return getattr(namespace, export["target"])
    return None


# Override __getattr__ to handle both submodules and direct class access
def __getattr__(name):
    """
    Lazily import modules or classes when accessed.

    This function enables direct access to classes like `scp.Coord`
    without importing them until they are actually used.
    """
    # Check plot profile functions first
    if name in _PLOT_PROFILE_FUNCTIONS:
        from spectrochempy.plotting import profile as _profile_module

        return getattr(_profile_module, name)

    # Ensure external plugins are discovered (before lazy imports so
    # plugin-provided functions take precedence over core stubs)
    plugin_manager.discover()

    import sys

    from spectrochempy.plugins.namespace import PluginNamespace
    from spectrochempy.plugins.namespace import PluginNamespaceModule
    from spectrochempy.plugins.namespace import has_namespace

    if has_namespace(registry, name):
        module_key = f"spectrochempy.{name}"
        if module_key in sys.modules and isinstance(
            sys.modules[module_key], PluginNamespaceModule
        ):
            return sys.modules[module_key]
        return PluginNamespace(name, plugin_manager, registry)

    # Check root-level compatibility aliases before extensions so that
    # deprecated root exports emit DeprecationWarning (the namespaced API
    # accessed via PluginNamespace does not warn).
    root_export = _plugin_root_export(name, PluginNamespace)
    if root_export is not None:
        return root_export

    # Check plugin readers first (e.g., read_topspin from external plugins)
    if name.startswith("read_"):
        reader_name = name[len("read_") :]
        reader_info = registry.get_reader(reader_name)
        if reader_info:
            return reader_info["func"]
        stub = plugin_reader_missing_stub(reader_name)
        if stub:
            return stub

    for category in ("analysis", "simulation", "accessor"):
        extension_info = registry.extensions.get(category, name)
        if extension_info:
            return extension_info["obj"]

    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[name])
        return getattr(module, name)

    # Look also NDDataset attribute which can be used as API methods
    if name in _LAZY_DATASETS_IMPORTS:
        from spectrochempy.core.dataset.nddataset import NDDataset

        return getattr(NDDataset, name)

    # Fall back to original __getattr__ for submodules
    try:
        return original_getattr(name)
    except AttributeError as err:
        # Check if this is a known plugin namespace
        hint = plugin_namespace_install_hint(name)
        if hint:
            from spectrochempy.plugins.deps import MissingPluginNamespaceError

            raise MissingPluginNamespaceError(hint) from err
        raise AttributeError(
            f"module 'spectrochempy' has no attribute '{name}'"
        ) from err


# --------------------------------------------------------------------------------------
# Register pseudo-modules for ``from spectrochempy.<ns> import X`` support
# --------------------------------------------------------------------------------------
from spectrochempy.plugins.namespace import register_namespace_modules

register_namespace_modules()
