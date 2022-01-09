# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""
import numpy as np
from datetime import datetime, timezone

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_jcamp", "write_jdx"]
__dataset_methods__ = __all__


# ...............................................................................
def write_jcamp(*args, **kwargs):
    """
    Write a dataset in JCAMP-DX format.

    (see Published JCAMP-DX Protocols http://www.jcamp-dx.org/protocols.html#ir4.24)
    Up to now, only IR output is available.

    Parameters
    ----------
    filename: str or pathlib objet, optional
        If not provided, a dialog is opened to select a file for writing.
    protocol : {'scp', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for writing. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        Where to write the specified `filename`. If not specified, write in the current directory.
    description: str, optional
        A Custom description.

    Returns
    -------
    out : `pathlib` object
        path of the saved file.

    Examples
    --------
    The extension will be added automatically
    >>> X.write_jcamp('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["JCAMP-DX files (*.jdx)"]
    kwargs["suffix"] = ".jdx"
    return exporter(*args, **kwargs)


write_jdx = write_jcamp
write_jdx.__doc__ = "This method is an alias of `write_jcamp`."


@exportermethod
def _write_jcamp(*args, **kwargs):
    # Writes a dataset in JCAMP-DX format

    dataset, filename = args
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
                if not title_index and type(label) is str:
                    title_index = i
                if not timestamp_index and type(label) is datetime:
                    timestamp_index = i

        if timestamp_index is None:
            timestamp = datetime.now(timezone.utc)

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

            fid.write(f'##LONGDATE={timestamp.strftime("%Y/%m/%d")}\n')
            fid.write(f'##TIME={timestamp.strftime("%H:%M:%S")}\n')

            fid.write("##XUNITS=1/CM\n")
            fid.write("##YUNITS=ABSORBANCE\n")

            firstx, lastx = dataset.x.data[0], dataset.x.data[-1]
            maxx, minx = max(firstx, lastx), min(firstx, lastx)
            xfactor = 1.0

            fid.write(f"##FIRSTX={firstx:.6f}\n")
            fid.write(f"##LASTX={lastx:.6f}\n")
            fid.write(f"##MAXX={maxx:.6f}\n")
            fid.write(f"##MINX={minx:.6f}\n")
            fid.write(f"##XFACTOR={xfactor}\n")

            firsty, lasty = dataset.data[0, 0], dataset.data[0, -1]
            # TODO : mask
            maxy, miny = np.nanmax(dataset.data), np.nanmin(dataset.data)
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
                    if np.isnan(dataset.data[i, j])
                    else f"{int(dataset.data[i, j] / yfactor):.6f} "
                )
                line += Y
                if len(line) >= 75 or j == nx - 1:
                    fid.write(f"{line}\n")
                    if j + 1 < nx:
                        line = f"{dataset.x.data[j + 1]:.6f} "

            fid.write("##END\n")

        fid.write("##END=" + "\n")

        return filename
