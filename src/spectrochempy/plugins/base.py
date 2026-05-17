# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SpectroChemPyPlugin(Protocol):
    name: str
    version: str
    PLUGIN_API_VERSION: str = "1.0"

    def register(self, registry) -> None:
        ...
