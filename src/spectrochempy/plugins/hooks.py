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

    @hookspec
    def plugin_info(self) -> dict:
        """
        Return plugin metadata.

        Expected return format::

            {
                "name": "my-plugin",
                "version": "0.1.0",
                "plugin_api_version": "1.0",
                "spectrochempy_min_version": "1.2",
                "description": "...",
                "capabilities": ["reader"],
            }

        Implementations that omit the method or return an empty dict
        are treated conservatively (no capability advertised).
        """
