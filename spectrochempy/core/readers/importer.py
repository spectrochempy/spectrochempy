#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
"""This module define a generic class to import files and contents.

"""
__all__ = ['read']
__dataset_methods__ = __all__

from warnings import warn
from datetime import datetime, timezone
from traitlets import HasTraits, List, Dict, Type, Unicode

from spectrochempy.utils import pathclean, check_filename_to_open, docstrings
from spectrochempy.utils.exceptions import DimensionsCompatibilityError, ProtocolError


# ----------------------------------------------------------------------------------------------------------------------
class Importer(HasTraits):
    # Private _Importer class

    objtype = Type
    datasets = List
    files = Dict
    default_key = Unicode
    protocol = Unicode

    protocols = Dict
    filetypes = Dict

    def __init__(self):

        FILETYPES = [
                ('scp', 'SpectroChemPy files (*.scp)'),
                ('omnic', 'Nicolet OMNIC files and series (*.spa *.spg *.srs)'),
                ('labspec', 'LABSPEC exported files (*.txt)'),
                ('opus', 'Bruker OPUS files (*.[0-9]*)'),
                ('topspin', 'Bruker TOPSPIN fid or series or processed data files (fid ser 1[r|i] 2[r|i]* 3[r|i]*)'),
                ('matlab', 'MATLAB files (*.mat)'),
                ('dso', 'Data Set Object files (*.dso)'),
                ('jcamp', 'JCAMP-DX files (*.jdx *.dx)'),
                ('csv', 'CSV files (*.csv)'),
                ('excel', 'Microsoft Excel files (*.xls)'),
                ('zip', 'Compressed folder of data files (*.zip)'),
                #  ('all', 'All files (*.*)')
                ]

        self.filetypes = dict(FILETYPES)
        temp = list(zip(*FILETYPES))
        temp.reverse()
        self.protocols = dict(zip(*temp))
        #  add alias
        ALIAS = [
                ('spg', 'omnic'),
                ('spa', 'omnic'),
                ('srs', 'omnic'),
                ('mat', 'matlab'),
                ('txt', 'labspec'),
                ('jdx', 'jcamp'),
                ('dx', 'jcamp'),
                ('xls', 'excel'),
                ]
        self.alias = dict(ALIAS)

    # ..................................................................................................................
    def __call__(self, *args, **kwargs):

        self.datasets = []
        self.default_key = kwargs.pop('default_key', '.scp')

        if 'merge' not in kwargs.keys():
            # if merge is not specified, but the args are provided as a single list, then will are supposed to merge
            # the datasets. If merge is specified then it has priority.
            # This is not usefull for the 1D datasets, as if they are compatible they are merged automatically
            if args and len(args) == 1 and isinstance(args[0], (list, tuple)):
                kwargs['merge'] = True

        args, kwargs = self._setup_objtype(*args, **kwargs)
        res = check_filename_to_open(*args, **kwargs)
        if res:
            self.files = res
        else:
            return None

        for key in self.files.keys():

            if key == 'frombytes':
                # here we need to read contents
                for filename, content in self.files[key].items():
                    files_ = check_filename_to_open(filename)
                    kwargs['content'] = content
                    key_ = list(files_.keys())[0]
                    self._switch_protocol(key_, files_, **kwargs)
                if len(self.datasets) > 1:
                    self.datasets = self._do_merge(self.datasets, **kwargs)

            else:
                # here files are read from the disk using filenames
                self._switch_protocol(key, self.files, **kwargs)

        if len(self.datasets) == 1:
            nd = self.datasets[0]  # a single dataset is returned
            name = kwargs.pop('name', None)
            if name:
                nd.name = name
            return nd

        else:
            nds = self.datasets
            names = kwargs.pop('names', None)
            if names and len(names) == len(nds):
                for nd, name in zip(nds, names):
                    nd.name = name
            elif names and len(names) != len(nds):
                warn('length of the `names` list and of the list of datsets mismatch - names not applied')
            return nds

    # ..................................................................................................................
    def _setup_objtype(self, *args, **kwargs):
        # check if the first argument is an instance of NDDataset, NDPanel, or Project

        args = list(args)
        if args and hasattr(args[0], 'implements') and args[0].implements() in ['NDDataset', 'NDPanel']:
            # the first arg is an instance of NDDataset or NDPanel
            object = args.pop(0)
            self.objtype = type(object)

        else:
            # by default returned objtype is NDDataset (import here to avoid circular import)
            from spectrochempy.core.dataset.nddataset import NDDataset

            self.objtype = kwargs.pop('objtype', NDDataset)

        return args, kwargs

    # ..................................................................................................................
    def _switch_protocol(self, key, files, **kwargs):

        protocol = kwargs.get('protocol', None)
        if protocol is not None and protocol != 'ALL':
            if not isinstance(protocol, list):
                protocol = [protocol]
            if key and key[1:] not in protocol and self.alias[key[1:]] not in protocol:
                return
        datasets = []
        for filename in files[key]:
            read_ = getattr(self, f"_read_{key[1:]}")
            try:
                res = read_(self.objtype(), filename, **kwargs)
                if not isinstance(res, list):
                    datasets.append(res)
                else:
                    datasets.extend(res)

            except NotImplementedError as e:
                raise e

            except IOError as e:
                if 'is not an Absorbance spectrum' in str(e):
                    # we do not read this filename
                    warn(str(e))
                else:
                    raise e

        if len(datasets) > 1:
            datasets = self._do_merge(datasets, **kwargs)
            if kwargs.get('merge', False):
                datasets[0].name = pathclean(filename).stem
                datasets[0].filename = pathclean(filename)

        self.datasets.extend(datasets)

    def _do_merge(self, datasets, **kwargs):

        # several datasets returned (only if several files have been passed) and the `merge` keyword argument is False
        merged = kwargs.get('merge', False)
        shapes = {nd.shape for nd in datasets}
        if len(shapes) == 1:
            # homogeneous set of files
            dim0 = shapes.pop()[0]
            if dim0 == 1:
                merged = kwargs.get('merge', True)  # priority to the keyword setting
        else:
            merged = kwargs.get('merge', False)
            # TODO: may be create a panel, when possible?

        if merged:
            # Try to stack the dataset into a single one
            try:
                dataset = self.objtype.stack(datasets)
                if kwargs.pop("sortbydate", True):
                    dataset.sort(dim='y', inplace=True)
                    dataset.history = str(datetime.now(timezone.utc)) + ':sorted by date'
                datasets = [dataset]

            except DimensionsCompatibilityError as e:
                warn(str(e))
                # return only the list

        return datasets


# ......................................................................................................................
def importermethod(func):
    # Decorateur
    setattr(Importer, func.__name__, staticmethod(func))
    return func


# ----------------------------------------------------------------------------------------------------------------------
# Generic Read function
# ----------------------------------------------------------------------------------------------------------------------

@docstrings.get_sections(base='read_method', sections=['Parameters', 'Other Parameters'])
@docstrings.dedent
def read(*paths, **kwargs):
    """
    Parameters
    ----------
    path : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False)
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True)
    description: str, optional
        A Custom description.
    origin : {'omnic', 'tga'}, optional
        in order to properly interpret CSV file it can be necessary to set the origin of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy `Preferences`.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory

    Other Parameters
    ----------------
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    Returns
    --------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) OPUS file(s).

    Examples
    ---------

    >>> from spectrochempy import NDDataset, read

    Reading a single OPUS file  (providing a windows type filename relative to the default ``Datadir``)

    >>> read('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the default ``Datadir``)
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> NDDataset.read('irdata/OPUS/test.0000')
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> read(p)
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is specified

    >>> l = read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(l)
    3
    >>> l[0]
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float32] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `merge`

    >>> read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float32] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> l = read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(l)
    3

    Read without a filename. This has the effect of opening a dialog for file(s) selection

    >>> nd = read()

    Read in a directory (assume that only OPUS files are present in the directory
    (else we must use the generic `read` function instead)

    >>> l = read(directory='irdata/OPUS')
    >>> len(l)
    4

    Again we can use merge to stack all 4 spectra if thet have compatible dimensions.

    >>> read(directory='irdata/OPUS', merge=True)
    NDDataset: [float32] a.u. (shape: (y:4, x:2567))

    See Also
    --------
    read_topspin: read TopSpin Bruker NMR spectra. Equivalent to set protocol='topspin'.
    read_omnic: read Omnic spectra
    read_spg: Read Omnic *.spg grouped spectra
    read_spa: Read Omnic *.Spa single spectra
    read_srs: Read Omnic series
    read_csv: Read CSV files
    read_zip: Read Zip files
    read_matlab: Read Matlab files

    """

    importer = Importer()

    protocol = kwargs.get('protocol', None)
    available_protocols = list(importer.protocols.values())
    available_protocols.extend(list(importer.alias.keys()))  # to handle variants of protocols
    if protocol is None:
        kwargs['filetypes'] = list(importer.filetypes.values())
        kwargs['protocol'] = 'ALL'
    else:
        try:
            kwargs['filetypes'] = [importer.filetypes[protocol]]
        except KeyError:
            raise ProtocolError(protocol, list(importer.protocols.values()))

    return importer(*paths, **kwargs)


# ......................................................................................................................
@importermethod
def _read_scp(*args, **kwargs):
    dataset, filename = args
    return dataset.load(filename, **kwargs)


# ......................................................................................................................
@importermethod
def _read_(*args, **kwargs):
    dataset, filename = args

    if not filename or filename.is_dir():
        return Importer._read_dir(*args, **kwargs)

    protocol = kwargs.get('protocol', None)
    if protocol and '.scp' in protocol:
        return dataset.load(filename, **kwargs)

    elif filename and filename.name in ('fid', 'ser', '1r', '2rr', '3rrr'):
        # probably an Topspin NMR file
        return dataset.read_topspin(filename, **kwargs)
    elif filename:
        # try scp format
        try:
            return dataset.load(filename, **kwargs)
        except Exception:
            # lets try some common format
            for key in ['omnic', 'opus', 'topspin', 'labspec', 'matlab', 'jdx']:
                try:
                    _read = getattr(dataset, f"read_{key}")
                    f = f'{filename}.{key}'
                    return _read(f, **kwargs)
                except Exception:
                    pass
            raise NotImplementedError


# ......................................................................................................................
docstrings.delete_params('read_method.parameters', 'origin', 'csv_delimiter')

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
