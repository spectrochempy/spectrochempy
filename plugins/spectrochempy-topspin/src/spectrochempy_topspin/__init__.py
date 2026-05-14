from spectrochempy.plugins.base import SpectroChemPyPlugin

from .read_topspin import read_topspin


class TopSpinPlugin(SpectroChemPyPlugin):
    name = "topspin"
    version = "0.1.0"
    api_version = "1.0"

    def register(self, registry) -> None:
        registry.register_reader(
            name="topspin",
            func=read_topspin,
            description="Bruker TOPSPIN fid or series or processed data files",
            extensions=[".fid", ".ser", "1r", "1i", "2rr", "2ri", "3rrr", "3rri"],
        )
