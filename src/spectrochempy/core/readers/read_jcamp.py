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
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.plugins.deps import MissingPluginNamespaceError
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.exceptions import SpectroChemPyError

# ======================================================================================
# Public functions
# ======================================================================================


def read_jcamp(*paths, **kwargs):
    r"""
    Open infrared, UV-visible, and supported processed 1D and real 2D NMR ``JCAMP-DX`` files with extension :file:`.jdx` or :file:`.dx`.

    This is the explicit JCAMP reader in the public import API. Use
    :func:`spectrochempy.read` for generic format autodetection and
    ``scp.jcamp.read(...)`` or :func:`spectrochempy.read_jcamp` when the JCAMP
    format is already known.

    Non-merged multi-file reads may return a list-like `ScpObjectList`
    exposing helper methods for dataset selection. See
    :func:`spectrochempy.read` for the complete description of the generic
    import convention and multi-object return behavior.

    The infrared reader supports AFFN and DIFDUP encoding
    (see :cite:t:`mcdonald:1988`).
    UV-visible spectra use explicit ASCII X/Y pairs in XYDATA or XYPOINTS.
    Processed NMR spectra use the optional NMR plugin's interop reader.

    Parameters
    ----------
    *paths : str or path-like, optional
        JCAMP-DX file paths, URLs, or a list of paths.
    jcamp_kind : {"auto", "ir", "nmr", "uvvis"}, optional
        Route by the declared data type by default. Set a kind explicitly
        when the declaration is ambiguous. Containers with conflicting
        declared spectrum types are rejected. Processed real 2D NMR spectra
        use the NTUPLES profile and require nmrglue-ng 1.3 or later.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s). When several datasets are returned, the
        result is a list-like `ScpObjectList` with helper attributes such as
        ``.names``, ``.select_largest()``, ``.select_by_name()``,
        ``.filter_by_ndim()``, and ``.filter_by_shape()``.

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a web application to handle drag and drop of files into a
        Browser.
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
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
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
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Using the explicit namespace API

    >>> scp.jcamp.read('irdata/jcamp/nh4y-activation.jdx')

    Using the legacy top-level alias

    >>> scp.read_jcamp('irdata/jcamp/nh4y-activation.jdx')

    """

    kwargs["filetypes"] = ["JCAMP-DX files (*.jdx *.dx)"]
    kwargs["protocol"] = ["jcamp"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# private functions
# ======================================================================================
def _routing_headers(stream):
    data_type = None
    for line in stream:
        line = line.strip()
        if not line.startswith("##"):
            if data_type is not None:
                data_type.append(line.partition("$$")[0])
            continue
        if data_type is not None:
            yield "##DATATYPE", "".join("".join(data_type).upper().split())
            data_type = None
        key, equal, value = line.partition("=")
        if not equal:
            continue
        key = re.sub(r"[\s\-_/]", "", key.upper())
        if key == "##DATATYPE":
            data_type = [value.partition("$$")[0]]
        elif key in {
            "##TITLE",
            "##END",
            "##XYDATA",
            "##XYPOINTS",
            "##PEAKTABLE",
            "##DATATABLE",
        }:
            yield key, ""
    if data_type is not None:
        yield "##DATATYPE", "".join("".join(data_type).upper().split())


def _declared_kind(stream, requested):
    if requested not in ("auto", "ir", "nmr", "uvvis"):
        raise SpectroChemPyError("jcamp_kind must be 'auto', 'ir', 'nmr' or 'uvvis'")

    spectrum_types = set()
    linked = False
    tables = 0
    tables_outside_2d = 0
    block_types = []
    for key, declared in _routing_headers(stream):
        if key == "##TITLE":
            block_types.append(None)
            continue
        if key == "##END":
            if block_types:
                block_types.pop()
            continue
        if key != "##DATATYPE":
            tables += 1
            tables_outside_2d += not block_types or block_types[-1] != "NDNMRSPECTRUM"
            continue
        if block_types:
            block_types[-1] = declared
        if declared == "LINK":
            linked = True
            continue
        spectrum_types.add(declared)
        if not linked and declared != "NDNMRSPECTRUM":
            break

    if len(spectrum_types) > 1:
        raise SpectroChemPyError("JCAMP-DX LINK contains conflicting data types")
    declared = next(iter(spectrum_types), None)
    if (
        declared is not None
        and declared.startswith("NMR")
        and declared != "NMRSPECTRUM"
    ):
        raise SpectroChemPyError(
            "JCAMP-DX routing supports processed 1D and real 2D NMR spectra only"
        )
    kind = {
        "INFRAREDSPECTRUM": "ir",
        "UVVISSPECTRUM": "uvvis",
        "UV/VISSPECTRUM": "uvvis",
        "NMRSPECTRUM": "nmr",
        "NDNMRSPECTRUM": "nmr",
    }.get(declared)
    dimension = "2d" if declared == "NDNMRSPECTRUM" else "1d"
    if kind is not None:
        if requested not in ("auto", kind):
            raise SpectroChemPyError(
                "jcamp_kind conflicts with the declared JCAMP-DX data type"
            )
        if dimension == "2d" and tables_outside_2d:
            raise SpectroChemPyError(
                "2D NMR JCAMP-DX contains tables outside its spectrum"
            )
        if kind == "nmr" and dimension == "1d" and linked and tables != 1:
            raise SpectroChemPyError(
                "NMR JCAMP-DX LINK requires exactly one spectral table"
            )
        return kind, linked, dimension
    if requested != "auto":
        return requested, linked, dimension
    if linked and not spectrum_types:
        return "ir", linked, dimension
    raise SpectroChemPyError(
        "Cannot determine JCAMP-DX spectrum type. Set jcamp_kind='ir', 'nmr' or 'uvvis'."
    )


def _read_nmr_jcamp(filename, content, dimension):
    import spectrochempy as scp  # noqa: PLC0415

    try:
        reader = scp.nmr.read
    except MissingPluginNamespaceError as exc:
        raise SpectroChemPyError(
            "NMR JCAMP-DX requires spectrochempy-nmr. "
            "Install it with pip install 'spectrochempy[nmr]'."
        ) from exc

    def read(path):
        try:
            return reader(
                path, protocol="interop", profile=f"jcampdx-spectrum-{dimension}"
            )
        except (ImportError, ValueError) as exc:
            raise SpectroChemPyError(
                f"Cannot read processed {dimension.upper()} NMR JCAMP-DX: {exc}"
            ) from exc

    if content is None:
        dataset = read(filename)
    else:
        with TemporaryDirectory(prefix="scpy-jcamp-") as temporary:
            path = Path(temporary) / Path(filename).name
            path.write_bytes(content)
            dataset = read(path)
    dataset.filename = filename
    dataset.origin = "jcamp-nmr"
    dataset.history = f"Imported NMR JCAMP-DX file {Path(filename).name}"
    return dataset


def _open_jcamp(filename, content, *, errors="strict"):
    if content is not None:
        return io.TextIOWrapper(
            io.BytesIO(content), encoding="utf-8-sig", errors=errors
        )
    return open(filename, encoding="utf-8-sig", errors=errors)  # noqa: SIM115


@_importer_method
def _read_jdx(dataset, filename, **kwargs):
    content = kwargs.get("content")
    with _open_jcamp(filename, content, errors="replace") as stream:
        kind, linked, dimension = _declared_kind(
            stream, kwargs.get("jcamp_kind", "auto")
        )
    if kind == "nmr":
        return _read_nmr_jcamp(filename, content, dimension)
    with _open_jcamp(filename, content) as stream:
        if kind == "uvvis":
            if linked:
                raise SpectroChemPyError(
                    "UV-visible JCAMP-DX requires a single spectrum"
                )
            return _read_uvvis_jcamp(dataset, filename, stream)
        return _read_ir_jcamp(
            dataset, filename, stream, linked, kwargs.get("sortbydate", True)
        )


# JCAMP-DX IR 4.24, section 5 and table VII.
_SQZ = dict(
    zip("@ABCDEFGHIabcdefghi", (0, *range(1, 10), *range(-1, -10, -1)), strict=True)
)
_DIF = dict(
    zip("%JKLMNOPQRjklmnopqr", (0, *range(1, 10), *range(-1, -10, -1)), strict=True)
)
_DUP = dict(zip("STUVWXYZs", range(1, 10), strict=True))
_DECIMAL_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_AFFN_NUMBER = rf"{_DECIMAL_NUMBER}(?:[Ee][+-]?[0-9]+)?"
_AFFN_ROW = re.compile(rf"{_AFFN_NUMBER}(?:\s+(?:{_AFFN_NUMBER}|\?))+")
_COMPRESSED_ROWS = tuple(
    re.compile(rf"({number})\s*([@A-Ia-i?].*)")
    for number in (_DECIMAL_NUMBER, _AFFN_NUMBER)
)
_PSEUDO_VALUE = re.compile(r"(?:[@A-Ia-i%J-Rj-rS-Zs][0-9]*|\?)")


def _read_ir_xydata(fid, firstx, lastx, npoints, xfactor):
    step = (lastx - firstx) / max(npoints - 1, 1)
    values = []
    checkpoint = False
    while True:
        keyword, text = _readl(fid)
        if keyword == "##END":
            break
        if keyword:
            raise ValueError(f"Unexpected JCAMP-DX record in XYDATA: {keyword}")
        if not text:
            continue
        if not checkpoint and _AFFN_ROW.fullmatch(text):
            values.extend(float(v) if v != "?" else np.nan for v in text.split()[1:])
            if len(values) > npoints:
                raise ValueError("JCAMP-DX XYDATA exceeds NPOINTS")
            continue

        index = len(values) - int(checkpoint)
        if index >= npoints:
            raise ValueError("JCAMP-DX XYDATA exceeds NPOINTS")
        # E/e can introduce either an exponent or a SQZ ordinate. Prefer the
        # closest grid position, with a compact SQZ ordinate first on ties.
        expected_x = firstx + index * step
        candidates = []
        for pattern in _COMPRESSED_ROWS:
            row = pattern.fullmatch(text)
            if row is None:
                continue
            anchor, encoded = row.groups()
            tokens = _PSEUDO_VALUE.findall(encoded)
            if "".join(tokens) != "".join(encoded.split()):
                continue
            # Integer anchors are commonly rounded or truncated by the exporter.
            precision = float(f"1e{Decimal(anchor).as_tuple().exponent}")
            anchor_x = float(anchor) * xfactor
            tolerance = abs(xfactor) * precision
            if (
                np.isfinite(anchor_x)
                and np.isfinite(tolerance)
                and np.isclose(anchor_x, expected_x, rtol=1e-12, atol=tolerance)
            ):
                candidates.append((abs(anchor_x - expected_x), tokens))
        if not candidates:
            raise ValueError("Invalid JCAMP-DX XYDATA row or abscissa")
        _, tokens = min(candidates, key=lambda candidate: candidate[0])
        difference = None
        previous = None
        for position, token in enumerate(tokens):
            prefix, tail = token[0], token[1:]
            if prefix in _DUP:
                if previous is None:
                    raise ValueError(
                        "JCAMP-DX DUP has no preceding ordinate or difference"
                    )
                repeats = int(f"{_DUP[prefix]}{tail}") - 1
                if len(values) + repeats > npoints:
                    raise ValueError("JCAMP-DX XYDATA exceeds NPOINTS")
                for _ in range(repeats):
                    values.append(
                        values[-1] + difference
                        if difference is not None
                        else values[-1]
                    )
                previous = None
                continue

            if prefix == "?":
                value = np.nan
                difference = None
            else:
                digits = _DIF if prefix in _DIF else _SQZ
                leading = digits[prefix]
                number = (-1 if leading < 0 else 1) * int(f"{abs(leading)}{tail}")
                difference = number if prefix in _DIF else None
                if difference is not None and (position == 0 or np.isnan(values[-1])):
                    raise ValueError(
                        "JCAMP-DX DIF requires a preceding absolute ordinate"
                    )
                value = values[-1] + difference if difference is not None else number
            if position == 0 and checkpoint:
                if value != values[-1]:
                    raise ValueError("JCAMP-DX DIF ordinate checkpoint does not match")
            else:
                values.append(value)
                if len(values) > npoints:
                    raise ValueError("JCAMP-DX XYDATA exceeds NPOINTS")
            previous = token if prefix != "?" else None
        checkpoint = difference is not None

    if checkpoint:
        raise ValueError("JCAMP-DX XYDATA is missing its final DIF checkpoint")
    if len(values) != npoints:
        raise ValueError("JCAMP-DX XYDATA does not match NPOINTS")
    return np.asarray(values, dtype=float)


def _read_uvvis_jcamp(dataset, filename, stream):
    headers = {}
    pairs = []
    reading = False
    while True:
        label, value = _readl(stream)
        label = re.sub(r"[\s\-_]", "", label.upper())
        value = value.strip()
        if label == "##END":
            break
        if label in {"##XYDATA", "##XYPOINTS"}:
            if reading or "".join(value.upper().split()) != "(XY..XY)":
                raise ValueError("UV-visible JCAMP-DX requires one explicit XY table")
            reading = True
        elif reading and value and not label:
            values = re.split(r"\s*[,;]\s*|\s+", value)
            if len(values) % 2 or not all(
                re.fullmatch(_AFFN_NUMBER, item) for item in values
            ):
                raise ValueError("Invalid UV-visible JCAMP-DX XY pair")
            pairs.extend(map(float, values))
        elif label:
            if reading or (label == "##TITLE" and label in headers):
                raise ValueError("UV-visible JCAMP-DX requires a single spectrum")
            headers[label] = value
    if any(line.partition("$$")[0].strip() for line in stream):
        raise ValueError("Unexpected content after UV-visible JCAMP-DX END")
    if not pairs or len(pairs) != 2 * int(headers.get("##NPOINTS", "0")):
        raise ValueError("UV-visible JCAMP-DX XY count disagrees with NPOINTS")
    xfactor = float(headers.get("##XFACTOR", "1"))
    yfactor = float(headers.get("##YFACTOR", "1"))
    with np.errstate(over="ignore", invalid="ignore"):
        x, y = np.asarray(pairs).reshape(-1, 2).T * np.array([[xfactor], [yfactor]])
    if xfactor == 0 or yfactor == 0 or not np.isfinite([x, y]).all():
        raise ValueError("Invalid UV-visible JCAMP-DX values or factors")
    for label, actual in (("##FIRSTX", x[0]), ("##LASTX", x[-1])):
        if label in headers and not np.isclose(
            float(headers[label]), actual, rtol=1e-7, atol=1e-12
        ):
            raise ValueError(f"UV-visible JCAMP-DX {label} disagrees with XY table")
    xunits = headers.get("##XUNITS", "")
    yunits = headers.get("##YUNITS", "")
    if xunits.casefold() not in {"nm", "nanometers", "wavelength (nm)"}:
        raise ValueError(f"Unsupported UV-visible JCAMP-DX XUNITS: {xunits!r}")
    coord = Coord(title="wavelength", units="nm", rounding=False)
    coord._set_data(x)
    dataset.data = y
    dataset.set_coordset(x=coord)
    dataset.x._rounding = False
    dataset.name = headers.get("##TITLE", Path(filename).stem)
    dataset.filename = filename
    dataset.origin = headers.get("##ORIGIN") or "jcamp-uvvis"
    dataset.title = yunits or "intensity"
    if yunits.casefold() in {"absorbance", "transmittance"}:
        dataset.units = yunits.lower()
    dataset.meta.jcamp_data_type = "UV/VIS SPECTRUM"
    dataset.meta.jcamp_xunits = xunits
    dataset.meta.jcamp_yunits = yunits
    dataset.history = f"Imported UV-visible JCAMP-DX file {Path(filename).name}"
    return dataset


def _read_ir_jcamp(dataset, filename, fid, linked, sortbydate):
    keyword = ""
    while keyword != "##TITLE":
        keyword, jdx_title = _readl(fid)

    nspec = 1
    if linked:
        while keyword != "##BLOCKS":
            keyword, text = _readl(fid)
        nspec = int(text)

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
        np.full(nspec, np.nan),
        np.full(nspec, np.nan),
    )

    # Read the spectra

    for i in range(nspec):
        # Reset variables
        keyword = ""
        xfactor = yfactor = 1.0
        values = None

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
                if (
                    not np.isfinite([firstx[i], lastx[i], xfactor, yfactor]).all()
                    or nx[i] <= 0
                    or xfactor == 0
                ):
                    raise ValueError(
                        "Invalid JCAMP-DX axis bounds, factors or point count"
                    )
                values = _read_ir_xydata(fid, firstx[i], lastx[i], nx[i], xfactor)
                spectra = (values * yfactor)[None, :]
                keyword = "##END"
                data = spectra if not data.size else np.concatenate((data, spectra), 0)

        if values is not None:
            if not xaxis.size:
                xaxis = np.linspace(firstx[i], lastx[i], nx[i])
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
            raise ValueError(f"Missing JCAMP-DX XYDATA in spectrum {i + 1}")

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

    dataset.data = data
    dataset.name = jdx_title
    dataset.filename = filename

    if yunits[0].strip() == "ABSORBANCE":
        dataset.units = "absorbance"
        dataset.title = "absorbance"
    elif yunits[0].strip() == "TRANSMITTANCE":
        dataset.units = "transmittance"
        dataset.title = "transmittance"
    elif yunits[0].strip() == "ARBITRARY UNITS":
        # The JCAMP token describes a unitless/arbitrary ordinate. Keep the
        # dataset units unset and do not synthesize a title from the token.
        dataset._units = None

    # now add coordinates
    _x = Coord(xaxis, title=axisname, units=axisunit, rounding=False, sigdigits=15)
    if linked:
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
    valid_dates = [date for date in alldates if date is not None]
    if valid_dates:
        dataset.acquisition_date = min(valid_dates)

    # Set origin, description and history
    origins = sorted({origin for origin in allorigins if origin})
    if len(origins) == 1:
        dataset.origin = origins[0]
    elif len(origins) > 1:
        dataset.origin = "; ".join(origins)

    dataset.description = f"Dataset from jdx file: '{jdx_title}'"

    dataset.history = f"Imported JCAMP-DX file {filename.name}"

    if sortbydate and nspec > 1:
        dataset.sort(dim="y", inplace=True)
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
        raise ValueError("Unexpected end of JCAMP-DX file")
    line = line.partition("$$")[0].strip()
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
