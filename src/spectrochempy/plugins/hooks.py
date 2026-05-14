# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import pluggy

hookspec = pluggy.HookspecMarker("spectrochempy")
hookimpl = pluggy.HookimplMarker("spectrochempy")


class SpectroChemPyHookSpec:
    @hookspec
    def get_filetype_info(self) -> dict:
        ...

    @hookspec
    def can_read(self, files: dict) -> bool:
        ...

    @hookspec
    def read_file(self, files: dict, protocol, **kwargs):
        ...
