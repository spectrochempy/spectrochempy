# ruff: noqa
import numpy as np

from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.params import Parameters

__docformat__ = "google"


class Data:
    """Class containing array data and associated parameter/metadata from an OPUS file.

    Args:
        data_block: parsed `FileBlock` instance of a data block
        data_status_block: `parsed FileBlock` instance of a data status block which contains metadata about the data
            block. This block is a parameter block.
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for the measurement (from param or rf_param block as appropriate)

    Attributes:
        params: `Parameter` class with metadata associated with the data block such as first x point: `fxp`, last x
            point: `lxp`, number of points: `npt`, date: `dat`, time: `tim` etc.
        y: 1D `numpy` array containing y values of data block
        x: 1D `numpy` array containing x values of data block. Units of x array are given by `dxu` parameter.
        label: human-readable string label describing the data block (e.g. Sample Spectrum, Absorbance, etc.)
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for the measurement (used to calculate modulation frequency)
        block: data `FileBlock` used to generate the `Data` class
        blocks: [data, data_status] `FileBlock` used to generate the `Data` class

    Extended Attributes:
        **wn:** Returns the x array in wavenumber (cm⁻¹) units regardless of what units the x array was originally
            saved in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **wl:** Returns the x array in wavelength (µm) units regardless of what units the x array was originally
            saved in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **f:** Returns the x array in modulation frequency units (Hz) regardless of what units the x array was
            originally saved in. This is only valid for spectral data blocks such as sample, reference, transmission,
            etc., not interferogram or phase blocks.
        **datetime:** Returns a `datetime` class of when the data was taken (extracted from data status parameter
            block).
        **xxx:** the various three char parameter keys from the `params` attribute can be directly called from the
            `Data` class for convenience. Common parameters include `dxu` (x units), `mxy` (max y value), `mny` (min y
            value), etc.
    """

    __slots__ = ("key", "params", "y", "x", "label", "vel", "block", "blocks")

    def __init__(
        self, data_block: FileBlock, data_status_block: FileBlock, key: str, vel: float
    ):
        self.key = key
        self.params = Parameters(data_status_block)
        if data_block.is_compact_data():
            y = data_block.data[-self.params.npt :]
        else:
            y = data_block.data
        self.y = (
            self.params.csf * y[: self.params.npt]
        )  # Trim extra values on some spectra
        self.x = np.linspace(self.params.fxv, self.params.lxv, self.params.npt)
        self.label = data_block.get_label()
        self.vel = vel
        data_block.data = None
        self.block = data_block
        self.blocks = self.params.blocks + [self.block]

    def __getattr__(self, name):
        if name.lower() == "wn" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_wn()
        if name.lower() == "wl" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_wl()
        if name.lower() == "f" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_freq()
        if name.lower() in self.params.keys():
            return getattr(self.params, name.lower())
        if name == "datetime":
            return self.params.datetime
        text = str(name) + " is not a valid attribute for Data: " + str(self.key)
        raise AttributeError(text)

    def _get_wn(self):
        if self.params.dxu == "WN":
            return self.x
        if self.params.dxu == "MI":
            return 10000.0 / self.x
        if self.params.dxu == "LGW":
            return np.exp(self.x)

    def _get_wl(self):
        if self.params.dxu == "WN":
            return 10000.0 / self.x
        if self.params.dxu == "MI":
            return self.x
        if self.params.dxu == "LGW":
            return 10000 / np.exp(self.x)

    def _get_freq(self):
        vel = 1000 * float(self.vel) / 7900  # cm/s
        return vel * self.wn


class DataSeries(Data):
    """Class containing a data series (3D specra) and associated parameter/metadata from an OPUS file.

    Args:
        data_block: parsed `FileBlock` instance of a data block
        data_status_block: `parsed FileBlock` instance of a data status block which contains metadata about the data
            block. This block is a parameter block.
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for measurement (from param or rf_param block as appropriate)

    Attributes:
        params: `Parameter` class with metadata associated with the data block such as first x point: `fxp`, last x
            point: `lxp`, number of points: `npt`, date: `dat`, time: `tim` etc.
        y: 2D numpy array containing y values of data block
        x: 1D numpy array containing x values of data block. Units of x array are given by `.dxu` attribute.
        num_spectra: number of spectra in the series (i.e. length of y)
        store_table: list if tuples that represent start and end indexes of saved spectra (useful for skipped spectra)
        label: human-readable string label describing the data block (e.g. Sample Spectrum, Absorbance, etc.)
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for measurement (used to calculate modulation frequency)
        block: data `FileBlock` used to generate the `DataSeries` class
        blocks: [data, data_status] `FileBlock` used to generate the `DataSeries` class

    Extended Attributes:
        **wn:** Returns the x array in wavenumber (cm⁻¹) units regardless of what units the x array was originally saved
            in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **wl:** Returns the x array in wavelength (µm) units regardless of what units the x array was originally saved
            in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **datetime:** Returns a `datetime` class of when the data was taken (extracted from data status parameter
            block).
        **xxx:** the various three char parameter keys from the `params` attribute can be directly called from the data
            class for convenience. Several of these parameters return arrays, rather than singular values because they
            are recorded for every spectra in the series, e.g. `npt`, `mny`, `mxy`, `srt`, 'ert', `nsn`.
    """

    __slots__ = (
        "key",
        "params",
        "y",
        "x",
        "label",
        "vel",
        "block",
        "blocks",
        "num_spectra",
        "version",
        "offset",
        "num_blocks",
        "data_size",
        "info_size",
        "store_table",
    )

    def __init__(
        self, data_block: FileBlock, data_status_block: FileBlock, key: str, vel: float
    ):
        self.key = key
        self.params = Parameters(data_status_block)
        data = data_block.data
        self.y = data["y"][:, : self.params.npt]  # Trim extra values on some spectra
        self.x = np.linspace(self.params.fxv, self.params.lxv, self.params.npt)
        self.num_spectra = len(self.y)
        for key in [
            "version",
            "offset",
            "num_blocks",
            "data_size",
            "info_size",
            "store_table",
        ]:
            setattr(self, key, data[key])
        for key, val in data.items():
            if key not in [
                "version",
                "offset",
                "num_blocks",
                "data_size",
                "info_size",
                "store_table",
            ]:
                self.params._params[key] = val
        self.label = data_block.get_label()
        self.vel = vel
        data_block.data = None
        self.block = data_block
        self.blocks = self.params.blocks + [self.block]
