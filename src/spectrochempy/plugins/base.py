# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SpectroChemPyPluginProtocol(Protocol):
    """
    Structural protocol for duck-type checks (e.g., ``isinstance``).

    This is **not** the base class for plugin authors.
    Use ``spectrochempy.api.plugins.SpectroChemPyPlugin`` (the concrete
    base class from ``spectrochempy.api.plugins``) when writing a plugin.
    """

    name: str
    version: str
    PLUGIN_API_VERSION: str = "1.0"

    def register(self, registry) -> None:
        ...
