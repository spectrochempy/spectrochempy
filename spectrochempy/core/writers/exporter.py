#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
"""
This module define a generic class to import files and contents.
"""
__all__ = ["write"]
__dataset_methods__ = __all__

from traitlets import HasTraits, Any

from spectrochempy.utils import pathclean, check_filename_to_save, patterns


# ------------------------------------------------------------------
class Exporter(HasTraits):
    # Exporter class

    object = Any

    def __init__(self):

        FILETYPES = [
            ("scp", "SpectroChemPy files (*.scp)"),
            ("labspec", "LABSPEC exported files (*.txt)"),
            ("matlab", "MATLAB files (*.mat)"),
            ("dso", "Data Set Object files (*.dso)"),
            ("jcamp", "JCAMP-DX files (*.jdx *dx)"),
            ("csv", "CSV files (*.csv)"),
            ("excel", "Microsoft Excel files (*.xls)"),
        ]

        self.filetypes = dict(FILETYPES)
        self.protocols = {}
        for protocol, filter in self.filetypes.items():
            for s in patterns(filter, allcase=False):
                self.protocols[s[1:]] = protocol

    # ..........................................................................
    def __call__(self, *args, **kwargs):

        args = self._setup_object(*args)

        try:
            if "filetypes" not in kwargs:
                kwargs["filetypes"] = list(self.filetypes.values())
                if args and args[0] is not None:  # filename
                    protocol = self.protocols[pathclean(args[0]).suffix]
                    kwargs["filetypes"] = [self.filetypes[protocol]]
            filename = check_filename_to_save(self.object, *args, **kwargs)
            if kwargs.get("suffix", ""):
                filename = filename.with_suffix(kwargs.get("suffix", ""))
            protocol = self.protocols[filename.suffix]
            write_ = getattr(self, f"_write_{protocol}")
            write_(self.object, filename, **kwargs)
            return filename

        except Exception as e:
            raise e

    # ..........................................................................
    def _setup_object(self, *args):

        # check if the first argument is an instance of NDDataset or Project
        args = list(args)

        if (
            args
            and hasattr(args[0], "implements")
            and args[0].implements() in ["NDDataset"]
        ):
            # the first arg is an instance of NDDataset
            self.object = args.pop(0)

        else:
            raise TypeError(
                "the API write method needs a NDDataset object as the first argument"
            )

        return args


# ..............................................................................
def exportermethod(func):
    # Decorator
    setattr(Exporter, func.__name__, staticmethod(func))
    return func


# ------------------------------------------------------------------
# Generic Read function
# ------------------------------------------------------------------
def write(dataset, filename=None, **kwargs):
    """
    Write  the current dataset.

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to write.
    filename : str or pathlib objet, optional
        If not provided, a dialog is opened to select a file for writing.
    **kwargs : dict
        See other parameters.

    Returns
    -------
    output_path
        Path of the output file.

    Other Parameters
    ----------------
    protocol : {'scp', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for writing. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        Where to write the specified `filename`. If not specified, write in the current directory.
    description: str, optional
        A Custom description.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy `Preferences`.

    See Also
    --------
    save : Generic function for saving a NDDataset in SpectroChemPy format.

    Examples
    ---------
    write a dataset (providing a windows type filename relative to the default ``Datadir``)

    >>> nd = scp.read_opus('irdata/OPUS')
    >>> f = nd.write('opus.scp')
    >>> f.name
    'opus.scp'
    """
    exporter = Exporter()
    return exporter(dataset, filename, **kwargs)


# ..............................................................................
@exportermethod
def _write_scp(*args, **kwargs):
    dataset, filename = args
    dataset.filename = filename
    return dataset.dump(filename, **kwargs)


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
