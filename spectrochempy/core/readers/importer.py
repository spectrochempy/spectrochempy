#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
"""This module define a generic class to import files and contents.

"""
__all__ = []

from warnings import warn
from datetime import datetime

from traitlets import HasTraits, List, Dict, Type, Unicode

from spectrochempy.utils import check_filename_to_open, docstrings, DimensionsCompatibilityError
from spectrochempy.core import error_


# Set some general parameters
# ----------------------------------------------------------------------------------------------------------------------
docstrings.get_sections(
        docstrings.dedent(
"""
Parameters
----------
*args : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
    The data source(s) can be specified by the name or a list of name for the file(s) to be loaded: 
    
    *e.g.,( file1, file2, ...,  **kwargs )*
    
    If the list of filenames are enclosed into brackets: 
    
    *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*
     
    The returned datasets are merged to form a single dataset, 
    except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`), 
    a dialog box will be opened to select files. 
directory : str, optional
    From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
    SpectroChemPy Preferences.
merge : bool, optional
    Default value is False. If True, and several filenames have been provided as arguments, 
    then a single dataset with merged (stacked along the first 
    dimension) is returned (default=False)
listdir : bool, optional
    If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge` 
    is True. It is assumed that all the files correspond to current reading protocol (default=True)
sortbydate : bool, optional
    Sort multiple spectra by acquisition date (default=True)
description: str, optional
    A Custom description.
origin : {'omnic', 'tga'}, optional
    in order to properly interpret CSV file it can be necessary to set the origin of the spectra.
    Up to now only 'omnic' and 'tga' have been implemented.
csv_delimiter : str, optional
    By default the column delimiter is the one set in SpectroChemPy `Preferences`.

Other Parameters
----------------
content : bytes object, optional
    Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
    The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application 
    to handle drag and drop of files into a Browser. 
    For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory 

"""
),
        base='read_method',
        sections=['Parameters', 'Other Parameters'])

docstrings.delete_params('read_method.parameters', 'origin', 'csv_delimiter')

class _Importer(HasTraits):
    # Private _Importer class

    datasets = List
    objtype = Type
    files = Dict
    default_key = Unicode
    protocol = Unicode

    def __call__(self, *args, **kwargs):

        from spectrochempy.core.dataset.nddataset import NDDataset

        self.datasets = []
        self.default_key = kwargs.pop('default_key','.spg')

        if 'objtype' not in kwargs.keys():
            kwargs['objtype'] = NDDataset

        if 'dictionary' not in kwargs.keys():
            kwargs['dictionary'] = True

        if 'merge' not in kwargs.keys():
            # if merge is not specified, but the args are provided as a single list, then will are supposed to merge
            # the datasets. If merge is specified then it has priority
            if args and len(args)==1 and isinstance(args[0], (list,tuple)):
                kwargs['merge'] = True

        res = check_filename_to_open(*args, **kwargs)
        if res:
            self.objtype, self.files = res
        else:
            return None

        for key in self.files.keys():

            if key != 'frombytes':
                # here files are read from the disk using filenames
                self._switch_protocol(key, self.files, **kwargs)
            else:
                # here we need to read contents
                for filename, content in self.files[key].items():
                    _, files_ = check_filename_to_open(filename)
                    kwargs['content'] = content
                    key_ = list(files_.keys())[0]
                    self._switch_protocol(key_, files_, **kwargs)

        if len(self.datasets) == 1:
            return self.datasets[0]  # a single dataset is returned

        # several datasets returned (only if several files have been passed) and the `merge` keyword argument is False
        merged = kwargs.get('merge', False)
        if not merged:
            return self.datasets
        else:
            # Try to stack the dataset into a single one
            try:
                dataset = self.objtype.stack(self.datasets)
            except DimensionsCompatibilityError as e:
                error_(str(e))
                return

            if kwargs.pop("sortbydate", True):
                dataset.sort(dim='y', inplace=True)
            dataset.history = str(datetime.now()) + ':sorted by date'
            return dataset

    def _switch_protocol(self, key, files, **kwargs):

        if not key:
            # default key if no key is specified
            key = self.default_key

        protocol = kwargs.get('protocol', None)
        if not isinstance(protocol, list):
            protocol = [protocol]
        if key not in protocol:
            return

        for filename in files[key]:
            read_ = getattr(self, f"_read_{key[1:]}")
            try:
                res = read_(self.objtype(), filename, **kwargs)
                if not isinstance(res, list):
                    self.datasets.append(res)
                else:
                    self.datasets.extend(res)
            except NotImplementedError as e:
                raise e
            except IOError as e:
                if 'is not an Absorbance spectrum' in str(e):
                    # we do not read this filename
                    warn(str(e))
                    continue
            except Exception as e:
                # try another format!
                self.datasets = self.objtype.read(self.datasets, files, protocol=key, **kwargs)
