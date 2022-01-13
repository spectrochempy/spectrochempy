# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = ["read_zip"]
__dataset_methods__ = __all__

import io

from spectrochempy.core.readers.importer import Importer, importermethod


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_zip(*paths, **kwargs):
    """
    Open a zipped list of data files.

    Parameters
    ----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_zip
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    origin : {'omnic', 'tga'}, optional
        in order to properly interpret CSV file it can be necessary to set the origin of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy ``Preferences``.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    --------

    >>> A = scp.read_zip('agirdata/P350/FTIR/FTIR.zip', only=50, origin='omnic')
    >>> print(A)
    NDDataset: [float64]  a.u. (shape: (y:50, x:2843))
    """
    kwargs["filetypes"] = ["Compressed files (*.zip)"]
    # TODO: allows other type of compressed files
    kwargs["protocol"] = ["zip"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================


@importermethod
def _read_zip(*args, **kwargs):
    # Below we assume that files to read are in a unique directory
    from spectrochempy.core.dataset.nddataset import NDDataset
    import zipfile

    # read zip file
    _, filename = args
    content = kwargs.pop("content", None)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")

    with zipfile.ZipFile(fid) as zf:

        filelist = zf.filelist
        only = kwargs.pop("only", len(filelist))

        datasets = []

        # for file in filelist:
        #
        #   # make a pathlib object (but this doesn't work with python 3.7)
        #     file = zipfile.Path(zf, at=file.filename)      # TODO:
        #
        #     if file.name.startswith('__MACOSX'):
        #         continue  # bypass non-data files
        #
        #     # seek the parent directory containing the files to read
        #     if not file.is_dir():
        #         continue
        #
        #     parent = file
        #     break
        #
        #
        # for count, children in enumerate(parent.iterdir()):
        #
        #     if count == only:
        #         # limits to only this number of files
        #         break
        #
        #     _ , extension = children.name.split('.')
        #     if extension == 'DS_Store':
        #         only += 1
        #         continue
        #
        #     read_ = getattr(NDDataset, f"read_{extension}")
        #
        #     datasets.append(read_(children.name, content=children.read_bytes(), **kwargs))

        # 3.7 compatible code

        # seek the parent directory containing the files to read
        for file in filelist:
            if not file.filename.startswith("__") and file.is_dir():
                parent = file.filename
                break

        count = 0
        for file in filelist:
            if (
                not file.is_dir()
                and file.filename.startswith(parent)
                and "DS_Store" not in file.filename
            ):
                # read it
                datasets.append(
                    NDDataset.read(
                        {file.filename: zf.read(file.filename)},
                        origin=kwargs.get("origin", None),
                    )
                )
                count += 1
                if count == only:
                    break
    return datasets


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
