# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["read_zip"]
__dataset_methods__ = __all__

from spectrochempy.core.readers.importer import (
    ALIAS,
    FILETYPES,
    Importer,
    _importer_method,
    _openfid,
    read,
)
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_zip")


@_docstring.dedent
def read_zip(*paths, **kwargs):
    """
    Open a zipped list of data files.

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
    %(Importer.see_also.no_read_zip)s

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

    # read zip file
    _, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

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

            # make a pathlib object (python > 3.7)
            file = zipfile.Path(zf, at=file.filename)
            # seek the parent directory containing the files to read
            if not file.is_dir():
                files.append(file)
            else:
                dirs.append(file)

        def extract(children, **kwargs):
            extension = children.name.split(".")[-1]
            if extension.lower() not in list(zip(*(ALIAS + FILETYPES)))[0]:
                return
            origin = kwargs.get("origin", "")
            return read(children.name, content=children.read_bytes(), origin=origin)

        # We assume that we have only a single dir with not subdir
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
                    d = extract(children, **kwargs)
                    if d is not None:
                        datasets.append(d)
                        count += 1
        else:
            for file in files:
                d = extract(file, **kwargs)
                if d is not None:
                    datasets.append(d)

        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets
