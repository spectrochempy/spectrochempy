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
