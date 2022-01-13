# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""
# import os as os
import csv

from spectrochempy.core import preferences as prefs
from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_csv"]
__dataset_methods__ = __all__


# ...............................................................................
def write_csv(*args, **kwargs):
    """
    Write a dataset in CSV format.

    Currently only implemented for 1D datasets
    or ND datasets with only one dimension of length larger than one.

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
    delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is ',' or the one set in SpectroChemPy `Preferences`.

    Returns
    -------
    out : `pathlib` object
        Path of the saved file.

    Examples
    --------

    >>> ds = scp.NDDataset([1, 2, 3])
    >>> f1 = ds.write_csv('myfile')

    >>> ds = scp.read('irdata/nh4y-activation.spg')
    >>> f2 = ds[0].write_csv('single_spectrum.csv')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["CSV files (*.csv)"]
    kwargs["suffix"] = ".csv"
    return exporter(*args, **kwargs)


@exportermethod
def _write_csv(*args, **kwargs):

    dataset, filename = args
    dataset.filename = filename

    delimiter = kwargs.get("delimiter", prefs.csv_delimiter)

    # check dimensionality of the dataset
    if dataset.squeeze().ndim > 1:
        raise NotImplementedError("Only implemented for 1D NDDatasets")

    # squeeze if necessary
    if dataset.ndim > 1:
        dataset = dataset.squeeze()

    # Make csv file for 1D dataset: first and 2d column are the unique axis and data, respectively
    with filename.open("w", newline="") as fid:
        writer = csv.writer(fid, delimiter=delimiter)

        if dataset.ndim == 1:  # if statement for future implementation for ndim > 1....
            if dataset.coordset is not None:
                col_coord = True
                title_1 = dataset.coordset[-1].title
                if dataset.coordset[-1].units is not None:
                    title_1 += " / " + str(dataset.coordset[-1].units)
            else:
                col_coord = False

            if dataset.units is not None:
                title_2 = dataset.title + " / " + str(dataset.units)
            else:
                title_2 = dataset.title

            if col_coord:
                coltitles = [title_1, title_2]
            else:
                coltitles = [title_2]

        writer.writerow(coltitles)
        if col_coord:
            for i, data in enumerate(dataset.data):
                writer.writerow([dataset.coordset[-1].data[i], data])
        else:
            for i, data in enumerate(dataset.data):
                writer.writerow([data])

    return filename
