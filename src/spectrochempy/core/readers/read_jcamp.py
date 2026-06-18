# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module to extend NDDataset with import methods."""

__all__ = ["read_jcamp"]

import io
import re
from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.decorators import deprecated

# ======================================================================================
# Public functions
# ======================================================================================


def read_jcamp(*paths, **kwargs):
    r"""
    Open Infrared ``JCAMP-DX`` files with extension :file:`.jdx` or :file:`.dx`.

    Limited to AFFN encoding (see :cite:t:`mcdonald:1988`)

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to ``False``.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or list of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application to handle drag and drop of files into a
        Browser.
    csv_delimiter : `str`, optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    description : `str`, optional
        A Custom description.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged dataset (stacked along the first
        dimension) is returned. In the case not all datasets have compatible dimensions or types/origins,
        then several NDDatasets can be returned for different groups of compatible datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used for instance whn reading directory with different types of files, for merging
        the datasets with compatible dimensions and different origin into different groups.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading, for example ``'scp'``, ``'omnic'``,
        ``'opus'``, ``'matlab'``, ``'jcamp'``, ``'csv'``, or ``'excel'``.
        If not provided, the correct protocol is inferred whenever possible
        from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    read_zip : Read Zip archives (containing spectrochempy readable files)
    read_dir : Read an entire directory.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra (:file:`.txt`).
    read_omnic : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    read_soc : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    read_galactic : Read Galactic files (:file:`.spc`).
    read_quadera : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.

    read_csv : Read CSV files (:file:`.csv`).
    read_matlab : Read Matlab files (:file:`.mat`, :file:`.dso`).
    read_wire : Read Renishaw Wire files (:file:`.wdf`).

    """

    kwargs["filetypes"] = ["JCAMP-DX files (*.jdx *.dx)"]
    kwargs["protocol"] = ["jcamp"]
    importer = Importer()
    return importer(*paths, **kwargs)


@deprecated(replace="read_jcamp", removed="0.11.0")
def read_jdx(*args, **kwargs):
    return read_jcamp(*args, **kwargs)


@deprecated(replace="read_jcamp", removed="0.11.0")
def read_dx(*args, **kwargs):  # pragma: no cover
    return read_jcamp(*args, **kwargs)


# ======================================================================================
# private functions
# ======================================================================================
@_importer_method
def _read_jdx(*args, **kwargs):
    # read jdx file
    dataset, filename = args
    content = kwargs.get("content")
    sortbydate = kwargs.pop("sortbydate", True)

    if content is not None:
        fid = io.StringIO(content.decode("utf-8"))
    else:
        fid = open(filename)  # noqa: SIM115

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
            if keyword == "##ORIGIN":
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
                    [allintensities],
                )  # convert allintensities into an array
                spectra[
                    spectra == "?"
                ] = "nan"  # deals with missing or out of range intensity values
                spectra = spectra.astype(np.float32)
                spectra *= yfactor
                # add spectra in "data" matrix
                data = spectra if not data.size else np.concatenate((data, spectra), 0)

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
                        "Inconsistent data set: number of wavenumber per spectrum should be identical",
                    )
                if firstx[i] - firstx[i - 1] != 0:
                    raise ValueError(
                        "Inconsistent data set: the x axis should start at same value",
                    )
                if lastx[i] - lastx[i - 1] != 0:
                    raise ValueError(
                        "Inconsistent data set: the x axis should end at same value",
                    )
        else:
            raise ValueError(
                f"##FIRSTX, ##LASTX or ##NPOINTS are unusable in spectrum n°{i + 1}"
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
                tzinfo=UTC,
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
                    f"##YUNITS should be the same for all spectra (check spectrum n°{i + 1}",
                )
            if xunits[i] != xunits[i - 1]:
                raise ValueError(
                    f"##XUNITS should be the same for all spectra (check spectrum n°{i + 1}",
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
    dataset.filename = filename

    if yunits[0].strip() == "ABSORBANCE":
        dataset.units = "absorbance"
        dataset.title = "absorbance"
    elif yunits[0].strip() == "TRANSMITTANCE":
        dataset.units = "transmittance"
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

    dataset.description = f"Dataset from jdx file: '{jdx_title}'"

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
            keyword, text = line.split("=", 1)
    else:
        keyword = ""
        text = line.strip()
    return keyword, text
