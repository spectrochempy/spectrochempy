# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plugin module to extend NDDataset with a JCAMP-DX export method."""

from datetime import datetime

import numpy as np

from spectrochempy.core.units import ur
from spectrochempy.core.writers.exporter import Exporter
from spectrochempy.core.writers.exporter import exportermethod
from spectrochempy.utils.datetimeutils import UTC

__all__ = ["write_jcamp"]
__dataset_methods__ = __all__


def write_jcamp(*args, **kwargs):
    """
    Write a dataset in JCAMP-DX format.

    (see Published JCAMP-DX Protocols http://www.jcamp-dx.org/protocols.html#ir4.24)
    Up to now, only IR output is available.

    Parameters
    ----------
    filename: str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing.
    directory : str, optional
        Where to write the specified `filename` . If not specified, write in the current directory.
    description: str, optional
        A Custom description.
    **kwargs
        Additional keyword arguments accepted by the generic writer API.
        This specialized writer always exports JCAMP-DX files.

    Returns
    -------
    out : `pathlib` object
        path of the saved file.

    Examples
    --------
    The extension will be added automatically
    >>> X.write_jcamp('myfile')

    Using the explicit namespace API
    >>> scp.jcamp.write(X, 'myfile')

    """
    exporter = Exporter()
    kwargs["filetypes"] = ["JCAMP-DX files (*.jdx)"]
    kwargs["suffix"] = ".jdx"
    return exporter(*args, **kwargs)


def _check_dataset_supported(dataset):
    """
    Validate that ``dataset`` matches the JCAMP-DX model this writer supports.

    The JCAMP writer requires a `y` coordinate describing the spectra axis and
    cannot represent complex data. Both checks run before any file is created
    or truncated and before any dataset attribute is modified.
    """
    try:
        y_coord = dataset.y
    except AttributeError:
        y_coord = None
    if y_coord is None:
        raise ValueError(
            "JCAMP export requires a `y` coordinate describing the spectra; "
            "the dataset has no `y` coordinate.",
        )
    if dataset.data.dtype.kind == "c":
        raise TypeError("JCAMP export does not support complex NDDataset data.")


def _format_unit(unit):
    return "None" if unit is None else str(unit)


def _unit_matches_exact_scale(unit, canonical_unit):
    """
    Return ``True`` only for exact-scale aliases of ``canonical_unit``.

    This intentionally relies on unit identity/equality, not generic
    dimensional compatibility or convertibility: values written by the JCAMP
    writer must not be implicitly rescaled.
    """
    return unit is not None and unit == ur.Unit(canonical_unit)


def _jcamp_xunits_token(dataset):
    x = dataset.x
    if x.unitless:
        return "ARBITRARY UNITS"
    if _unit_matches_exact_scale(x.units, "cm^-1"):
        return "1/CM"
    if _unit_matches_exact_scale(x.units, "um"):
        return "MICROMETERS"
    if _unit_matches_exact_scale(x.units, "nm"):
        return "NANOMETERS"
    raise ValueError(
        "JCAMP export only supports x coordinates with the exact numeric scale "
        "of cm^-1, um/µm, nm, or no unit. "
        f"Got x units {_format_unit(x.units)!r}. Convert explicitly before export.",
    )


def _jcamp_yunits_token(dataset):
    if dataset.unitless:
        return "ARBITRARY UNITS"
    if _unit_matches_exact_scale(dataset.units, "absorbance"):
        return "ABSORBANCE"
    if _unit_matches_exact_scale(dataset.units, "transmittance"):
        return "TRANSMITTANCE"
    raise ValueError(
        "JCAMP export only supports y units absorbance, transmittance, or no "
        f"unit. Got y units {_format_unit(dataset.units)!r}. Convert explicitly "
        "before export.",
    )


@exportermethod
def _write_jcamp(*args, **kwargs):
    # Writes a dataset in JCAMP-DX format

    dataset, filename = args
    _check_dataset_supported(dataset)
    xunits_token = _jcamp_xunits_token(dataset)
    yunits_token = _jcamp_yunits_token(dataset)
    dataset.filename = filename

    # Make JCAMP_DX file
    with filename.open("w") as fid:
        # Writes first lines
        fid.write(f"##TITLE={dataset.name}\n")
        fid.write("##JCAMP-DX=5.01\n")

        if dataset.shape[0] > 1:
            # Several spectra => Data Type = LINK
            fid.write("##DATA TYPE=LINK\n")
            # Number of spectra (size of 1st dimension)
            fid.write(f"##BLOCKS={dataset.shape[0]}\n")

        else:
            fid.write("##DATA TYPE=INFRARED SPECTRUM\n")

        # Determine whether the spectra have a title and a datetime field in the labels,
        # by default, the title if any will be is the first string; the timestamp will
        # be the fist datetime.datetime
        title_index = None
        timestamp_index = None
        if dataset.y.labels is not None:
            for i, label in enumerate(dataset.y.labels[0]):
                if not title_index and isinstance(label, str):
                    title_index = i
                if not timestamp_index and type(label) is datetime:
                    timestamp_index = i

        if timestamp_index is None:
            timestamp = datetime.now(UTC)

        # Masked values are exported as missing values: filling the mask with
        # NaN lets them flow through the same handling as genuine NaNs below
        # (excluded from MAXY/MINY and written as the JCAMP "?" marker).
        ydata = np.ma.filled(dataset.masked_data, np.nan)

        for i in range(dataset.shape[0]):
            if dataset.shape[0] > 1:
                title = (
                    dataset.y.labels[i][title_index]
                    if title_index
                    else f"spectrum #{i}"
                )
                fid.write(f"##TITLE={title}\n")
                fid.write("##JCAMP-DX=5.01\n")

            fid.write(f"##ORIGIN={dataset.origin}\n")
            fid.write(f"##OWNER={dataset.author}\n")

            if timestamp_index is not None:
                timestamp = dataset.y.labels[i][timestamp_index]

            fid.write(f"##LONGDATE={timestamp.strftime('%Y/%m/%d')}\n")
            fid.write(f"##TIME={timestamp.strftime('%H:%M:%S')}\n")

            fid.write(f"##XUNITS={xunits_token}\n")
            fid.write(f"##YUNITS={yunits_token}\n")

            firstx, lastx = dataset.x.data[0], dataset.x.data[-1]
            maxx, minx = max(firstx, lastx), min(firstx, lastx)
            xfactor = 1.0

            fid.write(f"##FIRSTX={firstx:.6f}\n")
            fid.write(f"##LASTX={lastx:.6f}\n")
            fid.write(f"##MAXX={maxx:.6f}\n")
            fid.write(f"##MINX={minx:.6f}\n")
            fid.write(f"##XFACTOR={xfactor}\n")

            spectrum = ydata[i]
            firsty, lasty = spectrum[0], spectrum[-1]
            maxy, miny = np.nanmax(spectrum), np.nanmin(spectrum)
            yfactor = 1.0e-8

            fid.write(f"##FIRSTY={firsty:.6f}\n")
            fid.write(f"##LASTY={lasty:.6f}\n")
            fid.write(f"##MAXY={maxy:.6f}\n")
            fid.write(f"##MINY={miny:.6f}\n")
            fid.write(f"##YFACTOR={yfactor}\n")

            nx = dataset.shape[1]
            fid.write(f"##NPOINTS={nx}\n")
            fid.write("##XYDATA=(X++(Y..Y))\n")

            line = f"{firstx:.6f} "
            for j in np.arange(nx):
                Y = (
                    "? "
                    if np.isnan(ydata[i, j])
                    else f"{int(ydata[i, j] / yfactor):.6f} "
                )
                line += Y
                if len(line) >= 75 or j == nx - 1:
                    fid.write(f"{line}\n")
                    if j + 1 < nx:
                        line = f"{dataset.x.data[j + 1]:.6f} "

            fid.write("##END\n")

        fid.write("##END=" + "\n")

        return filename
