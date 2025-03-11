import importlib

import numpy as np

from spectrochempy.core.dataset.baseobjects.ndcomplex import NDComplexArray
from spectrochempy.plugins.pluginmanager import SpectroChemPyPlugin
from spectrochempy.plugins.quaternion.core.dataset.baseobjects.ndquaternion import (
    NDQuaternionArray,
)


class QuaternionPlugin(SpectroChemPyPlugin):
    _auto_initialize = True

    @property
    def name(self) -> str:
        return "quaternion"

    @property
    def require_dependencies(self) -> list[str]:
        """Dict of required package dependencies with renaming."""
        return {"numpy-quaternion": "quaternion"}

    def initialize(self, manager) -> bool:
        """Initialize quaternion plugin."""
        if self._is_initialized:
            return True

        # Checks dependencies
        if not super().initialize(manager):
            return False

        # Import quaternion
        quaternion = importlib.import_module("quaternion")

        # Dynamically modify NDDataset's base class
        from spectrochempy.core.dataset.nddataset import NDDataset

        bases = list(NDDataset.__bases__)
        try:
            complex_idx = bases.index(NDComplexArray)
            bases[complex_idx] = NDQuaternionArray
            NDDataset.__bases__ = tuple(bases)
        except ValueError:
            # If NDComplexArray not in bases, append NDQuaternionArray
            NDDataset.__bases__ = (NDQuaternionArray,) + NDDataset.__bases__

        self._is_initialized = True
        return True

    def set_value(self, data, keys, value):
        """Handle quaternion value setting."""
        if np.isscalar(value):
            data[keys] = np.full_like(data[keys], value).astype(
                np.dtype(self.quaternion.quaternion)
            )
        else:
            data[keys] = value


plugin_class = QuaternionPlugin
