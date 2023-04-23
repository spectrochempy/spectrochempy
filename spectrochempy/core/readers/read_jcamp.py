# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module to extend NDDataset with the import methods.
"""

__all__ = ["read_jcamp"]
__dataset_methods__ = __all__

import io
import re
from datetime import datetime, timezone

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer, _importer_method
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_jcamp")


@_docstring.dedent
def read_jcamp(*paths, **kwargs):
    """
    Open Infrared ``JCAMP-DX`` files with extension :file:`.jdx` or :file:`.dx`\ .

    Limited to AFFN encoding (see :cite:t:`mcdonald:1988`\ )

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
    %(Importer.see_also.no_read_jcamp)s
    """
    kwargs["filetypes"] = ["JCAMP-DX files (*.jdx *.dx)"]
    kwargs["protocol"] = ["jcamp"]
    importer = Importer()
    return importer(*paths, **kwargs)


@deprecated(replace="read__jcamp")
def read_jdx(*args, **kwargs):
    return read_jcamp(*args, **kwargs)


@deprecated(replace="read_jcamp")
def read_dx(*args, **kwargs):  # pragma: no cover
    return read_jcamp(*args, **kwargs)


# ======================================================================================
# private functions
# ======================================================================================
@_importer_method
def _read_jdx(*args, **kwargs):

    # read jdx file
    dataset, filename = args
    content = kwargs.get("content", None)
    sortbydate = kwargs.pop("sortbydate", True)

    if content is not None:
        fid = io.StringIO(content.decode("utf-8"))
    else:
        fid = open(filename, "r")

    # Read header of outer Block

    keyword = ""

    while keyword != "##TITLE":
        keyword, text = _readl(fid)
    if keyword != "EOF":
        jdx_title = text
    else:  # pragma: no cover
        raise ValueError("No ##TITLE LR in outer block header")

    while (keyword != "##DATA TYPE") and (keyword != "##DATATYPE"):
        keyword, text = _readl(fid)
    if keyword != "EOF":
        jdx_data_type = text
    else:  # pragma: no cover
        raise ValueError("No ##DATA TYPE LR in outer block header")

    if jdx_data_type == "LINK":
        while keyword != "##BLOCKS":
            keyword, text = _readl(fid)
        nspec = int(text)
    elif jdx_data_type.replace(" ", "") == "INFRAREDSPECTRUM":
        nspec = 1
    else:
        raise ValueError("DATA TYPE must be LINK or INFRARED SPECTRUM")

    # Create variables

    xaxis = np.array([])
    data = np.array([])
    alltitles, alltimestamps, alldates, xunits, yunits, allorigins = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    nx, firstx, lastx = (
        np.zeros(nspec, "int"),
        np.zeros(nspec, "float"),
        np.zeros(nspec, "float"),
    )

    # Read the spectra

    for i in range(nspec):

        # Reset variables
        keyword = ""

        # (year, month,...) must be reset at each spectrum because labels "time"
        # and "longdate" are not required in JDX file
        [year, month, day, hour, minute, second] = "", "", "", "", "", ""

        # Read JDX file for spectrum n° i
        while keyword != "##END":
            keyword, text = _readl(fid)
            if keyword in ["##OWNER", "##JCAMP-DX"]:
                continue
            elif keyword == "##ORIGIN":
                allorigins.append(text)
            elif keyword == "##TITLE":
                # Add the title of the spectrum in the list alltitles
                alltitles.append(text)
            elif keyword == "##LONGDATE":
                [year, month, day] = text.split("/")
            elif keyword == "##TIME":
                [hour, minute, second] = re.split(r"[:.]", text)
            elif keyword == "##XUNITS":
                xunits.append(text)
            elif keyword == "##YUNITS":
                yunits.append(text)
            elif keyword == "##FIRSTX":
                firstx[i] = float(text)
            elif keyword == "##LASTX":
                lastx[i] = float(text)
            elif keyword == "##XFACTOR":
                xfactor = float(text)
            elif keyword == "##YFACTOR":
                yfactor = float(text)
            elif keyword == "##NPOINTS":
                nx[i] = float(text)
            elif keyword == "##XYDATA":
                # Read the intensities
                allintensities = []
                while keyword != "##END":
                    keyword, text = _readl(fid)
                    # for each line, get all the values except the first one (first value = wavenumber)
                    intensities = list(filter(None, text.split(" ")[1:]))
                    if len(intensities) > 0:
                        allintensities += intensities
                spectra = np.array(
                    [allintensities]
                )  # convert allintensities into an array
                spectra[
                    spectra == "?"
                ] = "nan"  # deals with missing or out of range intensity values
                spectra = spectra.astype(np.float32)
                spectra *= yfactor
                # add spectra in "data" matrix
                if not data.size:
                    data = spectra
                else:
                    data = np.concatenate((data, spectra), 0)

        # Check "firstx", "lastx" and "nx"
        if firstx[i] != 0 and lastx[i] != 0 and nx[i] != 0:
            if not xaxis.size:
                # Creation of xaxis if it doesn't exist yet
                xaxis = np.linspace(firstx[0], lastx[0], nx[0])
                xaxis = np.around((xaxis * xfactor), 3)
            else:
                # Check the consistency of xaxis
                if nx[i] - nx[i - 1] != 0:
                    raise ValueError(
                        "Inconsistent data set: number of wavenumber per spectrum should be identical"
                    )
                elif firstx[i] - firstx[i - 1] != 0:
                    raise ValueError(
                        "Inconsistent data set: the x axis should start at same value"
                    )
                elif lastx[i] - lastx[i - 1] != 0:
                    raise ValueError(
                        "Inconsistent data set: the x axis should end at same value"
                    )
        else:
            raise ValueError(
                "##FIRST, ##LASTX or ##NPOINTS are unusable in the spectrum n°", i + 1
            )

        # Creation of the acquisition date
        if (
            year != ""
            and month != ""
            and day != ""
            and hour != ""
            and minute != ""
            and second != ""
        ):
            date = datetime(
                int(year),
                int(month),
                int(day),
                int(hour),
                int(minute),
                int(second),
                tzinfo=timezone.utc,
            )
            timestamp = date.timestamp()
            # Transform back to timestamp for storage in the Coord object
            # use datetime.fromtimestamp(d, timezone.utc))
            # to transform back to datetime object
        else:
            timestamp = date = None
            # Todo: cases where incomplete date and/or time info
        alltimestamps.append(timestamp)
        alldates.append(date)

        # Check the consistency of xunits and yunits
        if i > 0:
            if yunits[i] != yunits[i - 1]:
                raise ValueError(
                    f"##YUNITS should be the same for all spectra (check spectrum n°{i + 1}"
                )
            elif xunits[i] != xunits[i - 1]:
                raise ValueError(
                    f"##XUNITS should be the same for all spectra (check spectrum n°{i + 1}"
                )

    # Determine xaxis name ****************************************************
    if xunits[0].strip() == "1/CM":
        axisname = "wavenumbers"
        axisunit = "cm^-1"
    elif xunits[0].strip() == "MICROMETERS":
        axisname = "wavelength"
        axisunit = "um"
    elif xunits[0].strip() == "NANOMETERS":
        axisname = "wavelength"
        axisunit = "nm"
    elif xunits[0].strip() == "SECONDS":
        axisname = "time"
        axisunit = "s"
    elif xunits[0].strip() == "ARBITRARY UNITS":
        axisname = "arbitrary unit"
        axisunit = None
    else:
        axisname = ""
        axisunit = ""
    fid.close()

    dataset.data = data
    dataset.name = jdx_title
    if yunits[0].strip() == "ABSORBANCE":
        dataset.units = "absorbance"
        dataset.title = "absorbance"
    elif yunits[0].strip() == "TRANSMITTANCE":
        # TODO: This units not in pint. Add this
        dataset.title = "transmittance"

    # now add coordinates
    _x = Coord(xaxis, title=axisname, units=axisunit)
    if jdx_data_type == "LINK":
        _y = Coord(
            alltimestamps,
            title="acquisition timestamp (GMT)",
            units="s",
            labels=(alldates, alltitles),
        )
        dataset.set_coordset(y=_y, x=_x)
    else:
        _y = Coord()
    dataset.set_coordset(y=_y, x=_x)

    # Set origin, description and history
    if nspec > 1:
        origins = set(allorigins)
        if len(origins) == 0:
            pass
        elif len(origins) == 1:
            dataset.origin = allorigins[0]
        else:
            dataset.origin = [(origin + "; ") for origin in set(allorigins)][0][:-2]

    dataset.description = "Dataset from jdx file: '{0}'".format(jdx_title)

    dataset.history = "Imported from jdx file"

    if sortbydate and nspec > 1:
        dataset.sort(dim="x", inplace=True)
        dataset.history = "Sorted by date"
    # Todo: make sure that the lowest index correspond to the largest wavenumber
    #  for compatibility with dataset created by read_omnic:

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset


@_importer_method
def _read_dx(*args, **kwargs):  # pragma: no cover
    return _read_jdx(*args, **kwargs)


def _readl(fid):
    line = fid.readline()
    if not line:
        return "EOF", ""
    line = line.strip(" \n")  # remove newline character
    if line[0:2] == "##":  # if line starts with "##"
        if line[0:5] == "##END":  # END KEYWORD, no text
            keyword = "##END"
            text = ""
        else:  # keyword + text
            keyword, text = line.split("=")
    else:
        keyword = ""
        text = line.strip()
    return keyword, text
