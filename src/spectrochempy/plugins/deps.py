# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


class MissingPluginError(ImportError):
    def __init__(
        self,
        feature: str,
        plugin_name: str = "spectrochempy-nmr",
        install_hint: str | None = None,
    ) -> None:
        msg = f"The '{feature}' feature requires the plugin '{plugin_name}'.\n"
        if install_hint:
            msg += install_hint
        else:
            msg += (
                f"Install it with : pip install {plugin_name}\n"
                f"Or using spectrochempy extras : pip install spectrochempy[{plugin_name.split('-')[-1]}]"
            )
        super().__init__(msg)


class PluginVersionError(RuntimeError):
    def __init__(self, plugin_name: str, required: str, found: str) -> None:
        super().__init__(
            f"Plugin '{plugin_name}' version {found} is not compatible. "
            f"Required version : {required}."
        )
