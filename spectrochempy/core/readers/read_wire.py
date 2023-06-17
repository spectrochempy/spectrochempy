# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
r"""
This module extend NDDataset with the import method for Labspec \*.txt generated data files.
"""

__all__ = ["read_wdf", "read_wire"]
__dataset_methods__ = __all__

import datetime

import numpy as np
from renishawWiRE import WDFReader

from spectrochempy.core.dataset.baseobjects.meta import Meta
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer, _importer_method, _openfid
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_wire")


@_docstring.dedent
def read_wire(*paths, **kwargs):
    """
    Read a single Raman spectrum or a series of Raman spectra.

    Files to open are :file:`.wdf` file created by Renishaw ``Wire`` software.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    --------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    ---------
    %(Importer.see_also.no_read_labspec)s
    """

    kwargs["filetypes"] = ["Renishaw WiRE files (*.wdf)"]
    kwargs["protocol"] = ["wire", "wdf"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_wdf = read_wire


# ======================================================================================
# Private Class
# ======================================================================================
class _WDFReader(WDFReader):
    # WDFReader is a class from the renishawWiRE package (MIT Licence - see
    # License in the root directory) It requires a filename
    # to be passed to the constructor. We want to be able to pass a file object.
    # So, we make a subclass of WDFReader, so we can modify the __init__ method.

    def __init__(self, file_obj):
        # This code is adapted from the WDFReader class (licence MIT),
        # except that we use the file_obj directly in the input.

        self.file_obj = file_obj

        # Initialize the properties for the wdfReader class
        self.title = ""
        self.username = ""
        self.measurement_type = None
        self.scan_type = None
        self.laser_length = None
        self.count = None
        self.spectral_unit = None
        self.xlist_type = None
        self.xlist_unit = None
        self.ylist_type = None
        self.ylist_unit = None
        self.point_per_spectrum = None
        self.data_origin_count = None
        self.capacity = None
        self.application_name = ""
        self.application_version = [None] * 4
        self.xlist_length = 0
        self.ylist_length = 0
        self.accumulation_count = None
        self.block_info = {}  # each key has value (uid, offset, size)
        self.is_completed = False
        self.debug = False

        # Parse the header section in the wdf file
        self.__locate_all_blocks()
        # Parse individual blocks
        self.__treat_block_data("WDF1")
        self.__treat_block_data("DATA")
        self.__treat_block_data("XLST")
        self.__treat_block_data("YLST")
        self.__treat_block_data("ORGN")
        self.__treat_block_data("WMAP")
        self.__treat_block_data("WHTL")

        # Reshape spectra after reading mapping information
        self.__reshape_spectra()


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_wdf(*args, **kwargs):
    # read WiRe *.wdf files or series

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    reader = _WDFReader(fid)

    meta = Meta()
    for item in (
        "title",
        "application_version",
        "laser_length",
        "count",
        "capacity",
        "accumulation_count",
        "point_per_spectrum",
        "scan_type",
        "measurement_type",
        "spectral_unit",
        "xlist_unit",
        "xlist_length",
        "ylist_unit",
        "ylist_length",
        "xpos_unit",
        "ypos_unit",
    ):
        try:
            meta[item] = getattr(reader, item)
        except AttributeError:
            pass

    # Get a single or serie of spectrum
    mtype = reader.measurement_type
    if mtype in [1, 2]:
        data = np.array(reader.spectra, ndmin=2)
        _x = Coord(
            reader.xdata, title=reader.xlist_unit.name, units=str(reader.xlist_unit)
        )
        if mtype == 1:
            _y = None
        else:
            # use the pos as y coordinate
            # Only a single position move is implemented since the matrix has 2D shape.
            idx = [
                ~np.all(reader.xpos == 0),
                ~np.all(reader.ypos == 0),
                ~np.all(reader.zpos == 0),
            ].index(True)
            position = (reader.xpos, reader.ypos, reader.zpos)[idx]
            u = getattr(reader, ("xpos_unit", "ypos_unit", "zpos_unit")[idx])
            _y = Coord(position, title="position", units=str(u))
        dataset.set_coordset(x=_x, y=_y)
        dataset.data = data
        dataset.units = str(reader.spectral_unit)
        dataset.title = reader.spectral_unit.name
        dataset.name = reader.title
        dataset.meta = meta
        dataset.history = f"Imported from {filename} on {datetime.datetime.now()}"

    # get mapped spectra
    elif mtype == 3:
        data = np.array(reader.spectra, ndmin=2)
        # X coordinate is still wavenumber
        _z = Coord(
            reader.xdata, title=reader.xlist_unit.name, units=str(reader.xlist_unit)
        )
        if data.ndim == 2:  # line mapped data
            # y coordinate is the position determined from xpos and ypos
            position = (reader.xpos**2 + reader.ypos**2) ** 0.5
            _y = Coord(position, title="distance", units=str(reader.xpos_unit))
            dataset.set_coordset(x=_z, y=_y)
        elif data.ndim == 3:  # grid mapped data
            grid = reader.map_shape
            xpos = reader.xpos.reshape(grid[::-1])[0]
            ypos = reader.ypos.reshape(grid[::-1])[:, 0]
            _x = Coord(xpos, title="x position", units=str(reader.xpos_unit))
            _y = Coord(ypos, title="y position", units=str(reader.ypos_unit))
            dataset.set_coordset(x=_z, y=_x, z=_y)

        dataset.data = data
        dataset.units = str(reader.spectral_unit)
        dataset.title = reader.spectral_unit.name
        dataset.name = reader.title
        dataset.meta = meta
        dataset.history = f"Imported from {filename} on {datetime.datetime.now()}"
        # reset modification date to cretion date
        dataset._modified = dataset._created
        dataset.filename = filename
    else:
        raise NotImplementedError(f"Measurement type {mtype} not implemented")

    return dataset
