# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["read_zip"]
__dataset_methods__ = __all__

import io

from spectrochempy.core.readers.importer import Importer, _importer_method


# ======================================================================================
# Public functions
# ======================================================================================
def read_zip(*paths, **kwargs):
    """
    Open a zipped list of data files.

    Parameters
    ----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        *e.g.,( file1, file2, ...,  \*\*kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, \*\*kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename` , nor `content` ),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read_zip
        `NDDataset` or list of `NDDataset` .

    Other Parameters
    ----------------
    protocol : 'str'\ , optional
        One of {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whenever it is possible) from the file name extension.
    directory : `str`\ , optional
        From where to read the specified `filename`\ . If not specified, read in the
        default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : `bool`, optional
        Default value is False. If True, and several filenames have been provided as
        arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : `bool`, optional
        Sort multiple spectra by acquisition date (default=True).
    description: `str`\ , optional
        A Custom description.
    origin : {'omnic', 'tga'}, optional
        in order to properly interpret CSV file it can be necessary to set the origin
        of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy `Preferences` .
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : `bool`, optional, default: `True`
        If `True` and filename is None, all files present in the provided `directory`
        are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol.
    recursive : bool, optional, default=False
        Read also in subfolders.

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic \*.spg grouped spectra.
    read_spa : Read Omnic \*.Spa single spectra.
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


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_zip(*args, **kwargs):
    # Below we assume that files to read are in a unique directory
    import zipfile

    from spectrochempy.core.dataset.nddataset import NDDataset

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

        files = []
        dirs = []

        for file in filelist:
            if "__MACOSX" in file.filename:
                continue  # bypass non-data files
            if ".DS_Store" in file.filename:
                continue

            # make a pathlib object (but this doesn't work with python 3.7)
            file = zipfile.Path(zf, at=file.filename)
            # seek the parent directory containing the files to read
            if not file.is_dir():
                files.append(file)
            else:
                dirs.append(file)

        def extract(children, **kwargs):
            # remove zip filetype and protocol
            # to use the one associated with the file extension
            origin = kwargs.get("origin", None)
            return NDDataset.read(
                children.name, content=children.read_bytes(), origin=origin
            )

        # we assume that only a single dir or a single file is zipped
        # But this can be changed later
        if dirs:
            # a single directory
            count = 0
            for children in dirs[0].iterdir():
                if count == only:
                    # limits to only this number of files
                    break
                elif "__MACOSX" in str(children.name):
                    continue  # bypass non-data files
                elif ".DS_Store" in str(children.name):
                    continue
                else:
                    # print(count, children)
                    # TODO: why this pose problem in pycharm-debug?????
                    datasets.append(extract(children, **kwargs))
                    count += 1

            return datasets
        else:
            return extract(files[0], **kwargs)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
