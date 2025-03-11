# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


class FileTypeRegistry:
    """Registry for file types and their handlers."""

    def __init__(self):
        self._filetypes: list[tuple[str, str]] = [
            ("scp", "SpectroChemPy files (*.scp)"),
            ("omnic", "Nicolet OMNIC files and series (*.spa *.spg *.srs)"),
            ("soc", "Surface Optics Corp. (*.ddr *.hdr *.sdr)"),
            ("labspec", "LABSPEC exported files (*.txt)"),
            ("opus", "Bruker OPUS files (*.[0-9]*)"),
            ("matlab", "MATLAB files (*.mat)"),
            ("dso", "Data Set Object files (*.dso)"),
            ("jcamp", "JCAMP-DX files (*.jdx *.dx)"),
            ("csv", "CSV files (*.csv)"),
            ("excel", "Microsoft Excel files (*.xls)"),
            ("zip", "Compressed folder of data files (*.zip)"),
            ("quadera", "Quadera ascii files (*.asc)"),
            ("carroucell", "Carroucell files (*spa)"),
            ("galactic", "GRAMS/Thermo Galactic files (*.spc)"),
            ("wire", "Renishaw WiRE files (*.wdf)"),
            (
                "topspin",
                "Bruker TOPSPIN fid or series or processed data files "
                "(fid ser 1[r|i] 2[r|i]* 3[r|i]*)",
            ),
        ]

        self._aliases: list[tuple[str, str]] = [
            ("spg", "omnic"),
            ("spa", "omnic"),
            ("ddr", "soc"),
            ("hdr", "soc"),
            ("sdr", "soc"),
            ("spc", "galactic"),
            ("srs", "omnic"),
            ("mat", "matlab"),
            ("txt", "labspec"),
            ("jdx", "jcamp"),
            ("dx", "jcamp"),
            ("xls", "excel"),
            ("asc", "quadera"),
            ("wdf", "wire"),
        ]
        self._exporttypes: list[tuple[str, str]] = [
            ("scp", "SpectroChemPy files (*.scp)"),
            ("labspec", "LABSPEC exported files (*.txt)"),
            ("matlab", "MATLAB files (*.mat)"),
            ("dso", "Data Set Object files (*.dso)"),
            ("jcamp", "JCAMP-DX files (*.jdx *dx)"),
            ("csv", "CSV files (*.csv)"),
            ("excel", "Microsoft Excel files (*.xls)"),
        ]

    def register_filetype(
        self, identifier: str, description: str, aliases: list[str] = None
    ):
        """Register a new file type."""
        self._filetypes.append((identifier, description))
        if aliases:
            for alias in aliases:
                self._aliases.append((alias, identifier))

    @property
    def filetypes(self) -> list[tuple[str, str]]:
        """Get all registered file types."""
        return self._filetypes.copy()

    @property
    def aliases(self) -> list[tuple[str, str]]:
        """Get all registered aliases."""
        return self._aliases.copy()

    @property
    def exporttypes(self) -> list[tuple[str, str]]:
        """Get all registered export types."""
        return self._exporttypes.copy()


# Global registry instance
registry = FileTypeRegistry()
