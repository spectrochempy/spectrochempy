# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


class MissingPluginError(ImportError):
    def __init__(
        self,
        feature: str,
        plugin_name: str = "",
        install_hint: str | None = None,
    ) -> None:
        if plugin_name:
            msg = f"The '{feature}' feature requires the plugin '{plugin_name}'.\n"
        else:
            msg = f"The '{feature}' feature requires an optional plugin.\n"
        if install_hint:
            msg += install_hint
        elif plugin_name:
            msg += (
                f"Install it with: pip install {plugin_name}\n"
                f"Or using spectrochempy extras: pip install spectrochempy[{plugin_name.split('-')[-1]}]"
            )
        super().__init__(msg)

    def _render_traceback_(self) -> list[str]:
        """Return a compact IPython/Jupyter rendering without the full traceback."""
        return [f"{type(self).__name__}: {self}\n"]


class MissingPluginNamespaceError(ImportError):
    """
    Error raised for missing official plugin namespaces.

    Using an ``ImportError`` allows ``from spectrochempy import iris`` to
    surface the actionable installation hint instead of Python's generic
    ``cannot import name`` message.
    """

    def _render_traceback_(self) -> list[str]:
        """Return a compact IPython/Jupyter rendering without the full traceback."""
        return [f"{type(self).__name__}: {self}\n"]


class PluginVersionError(RuntimeError):
    def __init__(self, plugin_name: str, required: str, found: str) -> None:
        super().__init__(
            f"Plugin '{plugin_name}' version {found} is not compatible. "
            f"Required version : {required}."
        )
