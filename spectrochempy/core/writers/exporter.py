#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
"""This module define a generic class to import files and contents.

"""
__all__ = ['write']
__dataset_methods__ = __all__

from warnings import warn

from traitlets import HasTraits, Any, List

from spectrochempy.utils import check_filename_to_save, docstrings
from spectrochempy.utils.qtfiledialogs import save_dialog


# ----------------------------------------------------------------------------------------------------------------------
class Exporter(HasTraits):
    # Exporter class

    object = Any
    filetypes = List(['SpectroChemPy files (*.scp)',
                      'MATLAB files (*.mat)',
                      'JCAMP-DX files (*.jdx)',
                      'CSV files (*.csv)',
                      'Microsoft Excel files (*.xls)',
                      'JSON format(*.json)'])
    suffixes = List(['.scp','.mat','.jdx','.csv','.xls', '.json'])
    protocols = List(['scp','matlab','jcamp','csv', 'excel', 'json'])

    def _suffix_from_protocol(self, val):
        d = dict(zip(self.protocols, self.suffixes))
        return d.get(val, '.scp')

    def _suffix_from_filetype(self, val):
        d = dict(zip(self.filetypes, self.suffixes))
        return d.get(val, '.scp')

    def _protocol_from_suffix(self, val):
        d = dict(zip(self.suffixes, self.protocols))
        return d.get(val, 'scp')

    # ..................................................................................................................
    def __call__(self, *args, **kwargs):

        args = self._setup_object(*args, **kwargs)

        try:
            if 'filetypes' not in kwargs:
                kwargs['filetypes'] = self.filetypes
                kwargs['suffix']='.scp'
            filename = check_filename_to_save(self.object, *args, **kwargs)
            protocol = self._protocol_from_suffix(filename.suffix)
            write_ = getattr(self, f"_write_{protocol}")
            write_(self.object, filename, **kwargs)
            return filename

        except Exception as e:
            raise e

    # ..................................................................................................................
    def _setup_object(self, *args, **kwargs):

        # check if the first argument is an instance of NDDataset, NDPanel, or Project
        args = list(args)

        if args and hasattr(args[0], 'implements') and args[0].implements() in ['NDDataset', 'NDPanel']:
            # the first arg is an instance of NDDataset or NDPanel
            self.object = args.pop(0)

        else:
            raise TypeError('the API write method needs a NDDataset or NDPanel object as the first argument')

        return args


# ......................................................................................................................
def exportermethod(func):
    # Decorator
    setattr(Exporter, func.__name__, staticmethod(func))
    return func


# ----------------------------------------------------------------------------------------------------------------------
# Generic Read function
# ----------------------------------------------------------------------------------------------------------------------

@docstrings.get_sections(base='write_method', sections=['Parameters', 'Other Parameters'])
@docstrings.dedent
def write(*args, **kwargs):
    """
    Parameters
    ----------
    filename: str or pathlib objet, optional
        If not provided, a dialog is opened to select a file for writing
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


    Examples
    ---------

    >>> from spectrochempy import read_opus, write

    write a dataset (providing a windows type filename relative to the default ``Datadir``)

    >>> nd = read_opus('irdata/OPUS')
    >>> nd.write('irdata/mydataset.scp')


    See Also
    --------
    save : generic function for saving a NDDataset in SpectroChemPy format
    write_spc, write_json, write_matlab, write_jcamp, write_excel, write_csv

    """
    exporter = Exporter()
    return exporter(*args, **kwargs)


# ......................................................................................................................
@exportermethod
def _write_scp(*args, **kwargs):
    dataset, filename = args
    dataset.filename = filename
    return dataset._save(filename)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
