# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Stable plugin base class.

Plugins should inherit from ``SpectroChemPyPlugin`` and override
``register()`` to declare their capabilities.
"""

from typing import Any

from spectrochempy.api.plugins.constants import CORE_PLUGIN_API_VERSION


class SpectroChemPyPlugin:
    """
    Base class for all SpectroChemPy plugins.

    Subclasses set class-level metadata and implement ``register()``
    to register readers, writers or processors via the provided helper
    methods (``_register_reader``, ``_register_writer``, etc.).
    """

    PLUGIN_API_VERSION: str = CORE_PLUGIN_API_VERSION

    name: str = "unnamed"
    version: str = "0.0.0"
    description: str = ""
    spectrochempy_min_version: str = "0.0.0"

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register(self, registry: Any) -> None:
        """
        Register plugin components into the system.

        Subclasses **must** override this method.  Use the helper
        methods below instead of calling ``registry`` directly so that
        future declarative hook support is transparent.

        Parameters
        ----------
        registry : PluginRegistry
            The central plugin registry singleton.
        """

    def _register_reader(
        self,
        registry: Any,
        name: str,
        func: Any,
        *,
        description: str = "",
        extensions: list[str] | None = None,
    ) -> None:
        """Register a file-reader."""
        registry.register_reader(
            name, func, description=description, extensions=extensions
        )

    def _register_writer(
        self,
        registry: Any,
        name: str,
        func: Any,
        *,
        description: str = "",
    ) -> None:
        """Register a file-writer."""
        registry.register_writer(name, func, description=description)

    def _register_processor(
        self,
        registry: Any,
        name: str,
        func: Any,
        *,
        description: str = "",
    ) -> None:
        """Register a processing function."""
        registry.register_processor(name, func, description=description)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def plugin_info(self) -> dict[str, Any]:
        """
        Return plugin metadata as a dict.

        The returned dict is used by the compatibility validation
        machinery.  Subclasses may override to add custom keys.
        """
        return {
            "name": self.name,
            "version": self.version,
            "plugin_api_version": self.PLUGIN_API_VERSION,
            "spectrochempy_min_version": self.spectrochempy_min_version,
            "description": self.description,
        }
