# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module define a generic class to import directories, files and contents.
"""
__all__ = ["read", "read_dir", "read_remote"]
__dataset_methods__ = __all__

from io import BytesIO
from warnings import warn
from zipfile import ZipFile

import requests
import yaml
from traitlets import Dict, HasTraits, List, Type, Unicode

from spectrochempy.core import info_, warning_
from spectrochempy.utils import (
    check_filename_to_open,
    get_directory_name,
    get_filenames,
    pathclean,
)
from spectrochempy.utils.exceptions import DimensionsCompatibilityError, ProtocolError

FILETYPES = [
    ("scp", "SpectroChemPy files (*.scp)"),
    ("omnic", "Nicolet OMNIC files and series (*.spa *.spg *.srs)"),
    ("soc", "Surface Optics Corp. (*.ddr *.hdr *.sdr)"),
    ("labspec", "LABSPEC exported files (*.txt)"),
    ("opus", "Bruker OPUS files (*.[0-9]*)"),
    (
        "topspin",
        "Bruker TOPSPIN fid or series or processed data files (fid ser 1[r|i] 2[r|i]* 3[r|i]*)",
    ),
    ("matlab", "MATLAB files (*.mat)"),
    ("dso", "Data Set Object files (*.dso)"),
    ("jcamp", "JCAMP-DX files (*.jdx *.dx)"),
    ("csv", "CSV files (*.csv)"),
    ("excel", "Microsoft Excel files (*.xls)"),
    ("zip", "Compressed folder of data files (*.zip)"),
    ("quadera", "Quadera ascii files (*.asc)"),
    ("carroucell", "Carroucell files (*spa)"),
    ("galactic", "GRAMS/Thermo Galactic files (*.spc)")
    #  ('all', 'All files (*.*)')
]
ALIAS = [
    ("spg", "omnic"),
    ("spa", "omnic"),
    ("ddr", "soc"),
    ("hdr", "soc"),
    ("sdr", "soc"),
    ("spc", "galactic"),
    ("srs", "omnic"),
    ("mat", "matlab"),
    ("txt", "labspec"),
    ("jdx", "jcamp"),
    ("dx", "jcamp"),
    ("xls", "excel"),
    ("asc", "quadera"),
]


# ------------------------------------------------------------------
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

        super().__init__()

        self.filetypes = dict(FILETYPES)
        temp = list(zip(*FILETYPES))
        temp.reverse()
        self.protocols = dict(zip(*temp))

        #  add alias

        self.alias = dict(ALIAS)

    def __call__(self, *args, **kwargs):

        self.datasets = []
        self.default_key = kwargs.pop("default_key", ".scp")

        if "merge" not in kwargs.keys():
            # if merge is not specified, but the args are provided as a single list, then will are supposed to merge
            # the datasets. If merge is specified then it has priority.
            # This is not useful for the 1D datasets, as if they are compatible they are merged automatically
            if args and len(args) == 1 and isinstance(args[0], (list, tuple)):
                kwargs["merge"] = True

        args, kwargs = self._setup_objtype(*args, **kwargs)
        res = check_filename_to_open(*args, **kwargs)
        if res:
            # Normal return
            self.files = res
        else:
            # Cancel in dialog!
            return None

        for key in self.files.keys():

            if key == "" and kwargs.get("protocol") == ["carroucell"]:
                key = ".carroucell"
                self.files = {".carroucell": self.files[""]}

            if key == "frombytes":
                # here we need to read contents
                for filename, content in self.files[key].items():
                    files_ = check_filename_to_open(filename)
                    kwargs["content"] = content
                    key_ = list(files_.keys())[0]
                    self._switch_protocol(key_, files_, **kwargs)
                if len(self.datasets) > 1:
                    self.datasets = self._do_merge(self.datasets, **kwargs)

            elif key and key[1:] not in list(zip(*FILETYPES))[0] + list(zip(*ALIAS))[0]:
                raise TypeError(f"Filetype `{key}` is unknown in spectrochempy")
            else:
                # here files are read / or remotely from the disk using filenames
                self._switch_protocol(key, self.files, **kwargs)

        # now we will reset preference for this newly loaded datasets
        if len(self.datasets) > 0:

            if all(self.datasets) is None:
                return None

            prefs = self.datasets[0].preferences
            try:
                prefs.reset()
            except (FileNotFoundError, AttributeError):
                pass
        else:
            return None

        if len(self.datasets) == 1:
            nd = self.datasets[0]  # a single dataset is returned
            name = kwargs.pop("name", None)
            if name:
                nd.name = name
            return nd

        else:
            nds = self.datasets
            names = kwargs.pop("names", None)
            if names and len(names) == len(nds):
                for nd, name in zip(nds, names):
                    nd.name = name
            elif names and len(names) != len(nds):
                warn(
                    "length of the `names` list and of the list of datasets mismatch - names not applied"
                )
            return sorted(
                nds, key=str
            )  # return a sorted list (sorted according to their string representation)

    def _setup_objtype(self, *args, **kwargs):
        # check if the first argument is an instance of NDDataset or Project

        args = list(args)
        if (
            args
            and hasattr(args[0], "implements")
            and args[0].implements() in ["NDDataset"]
        ):
            # the first arg is an instance of NDDataset
            object = args.pop(0)
            self.objtype = type(object)

        else:
            # by default returned objtype is NDDataset (import here to avoid circular import)
            from spectrochempy.core.dataset.nddataset import NDDataset

            self.objtype = kwargs.pop("objtype", NDDataset)

        return args, kwargs

    def _switch_protocol(self, key, files, **kwargs):

        protocol = kwargs.get("protocol", None)
        if protocol is not None and protocol != "ALL":
            if not isinstance(protocol, list):
                protocol = [protocol]
            if key and key[1:] not in protocol and self.alias[key[1:]] not in protocol:
                return
        datasets = []
        for filename in files[key]:
            filename = pathclean(filename)
            read_ = getattr(self, f"_read_{key[1:]}")
            try:
                res = read_(self.objtype(), filename, **kwargs)
                # sometimes read_ can return None (e.g. non labspec text file)
            except FileNotFoundError as e:
                # try to get the file from github
                kwargs["read_method"] = read_
                try:
                    res = _read_remote(self.objtype(), filename, **kwargs)

                except OSError:
                    raise e

                except IOError as e:
                    warning_(str(e))
                    res = None

                except NotImplementedError as e:
                    warning_(str(e))
                    res = None

                except Exception:
                    raise e

            except IOError as e:
                warning_(str(e))
                res = None

            except NotImplementedError as e:
                warning_(str(e))
                res = None

            if res is not None:
                if not isinstance(res, list):
                    datasets.append(res)
                else:
                    datasets.extend(res)

        if len(datasets) > 1:
            datasets = self._do_merge(datasets, **kwargs)
            if kwargs.get("merge", False):
                datasets[0].name = pathclean(filename).stem
                datasets[0].filename = pathclean(filename)

        self.datasets.extend(datasets)

    def _do_merge(self, datasets, **kwargs):

        # several datasets returned (only if several files have been passed) and the `merge` keyword argument is False
        merged = kwargs.get("merge", False)
        shapes = {nd.shape for nd in datasets}
        if len(shapes) == 1:
            # homogeneous set of files
            dim0 = shapes.pop()[0]
            if dim0 == 1:
                merged = kwargs.get("merge", True)  # priority to the keyword setting
        else:
            # not homogeneous
            merged = kwargs.get("merge", False)

        if merged:
            # Try to stack the dataset into a single one
            try:
                dataset = self.objtype.concatenate(datasets, axis=0)
                if dataset.coordset is not None and kwargs.pop("sortbydate", True):
                    dataset.sort(dim="y", inplace=True)
                    dataset.history = "Sorted by date"
                datasets = [dataset]

            except DimensionsCompatibilityError as e:
                warn(str(e))  # return only the list

        return datasets


def _importer_method(func):
    # Decorator to define a given read function as belonging to Importer
    setattr(Importer, func.__name__, staticmethod(func))
    return func


# ------------------------------------------------------------------
# Generic Read function
# ------------------------------------------------------------------
def read(*paths, **kwargs):
    """
    Generic read method.

    This method is generally able to load experimental files based on extensions.

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
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whenever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description : str, optional
        A Custom description.
    origin : {'omnic', 'tga'}, optional
        In order to properly interpret CSV file it can be necessary to set the origin of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy ``Preferences``.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_spc : Read Galactic *.spc files.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    ---------
    Reading a single OPUS file  (providing a windows type filename relative to the default ``Datadir``)

    >>> scp.read('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the default ``Datadir``)
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> scp.NDDataset.read('irdata/OPUS/test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> scp.read(p)
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is specified

    >>> le = scp.read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `merge`

    >>> scp.read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(le)
    3

    Read without a filename. This has the effect of opening a dialog for file(s) selection

    >>> nd = scp.read()

    Read in a directory (assume that only OPUS files are present in the directory
    (else we must use the generic `read` function instead)

    >>> le = scp.read(directory='irdata/OPUS')
    >>> len(le)
    2

    Again we can use merge to stack all 4 spectra if thet have compatible dimensions.

    >>> scp.read(directory='irdata/OPUS', merge=True)
    [NDDataset: [float64] a.u. (shape: (y:1, x:5549)), NDDataset: [float64] a.u. (shape: (y:4, x:2567))]
    """

    importer = Importer()

    protocol = kwargs.get("protocol", None)
    available_protocols = list(importer.protocols.values())
    available_protocols.extend(
        list(importer.alias.keys())
    )  # to handle variants of protocols
    if protocol is None:
        kwargs["filetypes"] = list(importer.filetypes.values())
        kwargs["protocol"] = "ALL"
        default_filter = kwargs.get("default_filter", None)
        if default_filter is not None:
            kwargs["default_filter"] = importer.filetypes[default_filter]
    else:
        try:
            kwargs["filetypes"] = [importer.filetypes[protocol]]
        except KeyError:
            raise ProtocolError(protocol, list(importer.protocols.values()))

    return importer(*paths, **kwargs)


def read_dir(directory=None, **kwargs):
    """
    Read an entire directory.

    Open a list of readable files in a and store data/metadata in a dataset or a list of datasets according to the
    following rules :

    * 2D spectroscopic data (e.g. valid *.spg files or matlab arrays, etc...) from
      distinct files are stored in distinct NDdatasets.
    * 1D spectroscopic data (e.g., *.spa files) in a given directory are grouped
      into single NDDataset, providing their unique dimension are compatible. If not,
      an error is generated.
    * non readable files are ignored

    Parameters
    ----------
    directory : str or pathlib
        Folder where are located the files to read.

    Returns
    --------
    read_dir
        |NDDataset| or list of |NDDataset|.

    Depending on the python version, the order of the datasets in the list may change.

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    --------

    >>> scp.preferences.csv_delimiter = ','
    >>> A = scp.read_dir('irdata')
    >>> len(A)
    4

    >>> B = scp.NDDataset.read_dir()
    """
    kwargs["listdir"] = True
    importer = Importer()
    return importer(directory, **kwargs)


def read_remote(file_or_dir, **kwargs):
    """
    Download and read files or an entire directory on github spectrochempy_data
    repository.

    This is done only if the data are not yet downloaded and present in the DATADIR
    directory.

    Parameters
    ----------
    file_or_dir : str or pathlib
        Folder where are located the files to read
        (it should be written as if it was locally downloaded already).
    **kwargs : optional keywords parameters
        See Other Parameters.

    Other Parameters
    ----------------
    merge : bool, optional, default: False
        By default files read are noot merged
    replace_existing: bool, optional, default: False
        By default, existing files are not replaced so not downloaded.
    download_only: bool, optional, default: False
        If True, only downloading and saving of the files is performed, with no
        attempt to read their content.

    Returns
    --------
    dataset(s)
        |NDDataset| or list of |NDDataset|.

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic \*.spg grouped spectra.
    read_spa : Read Omnic \*.spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    --------

    >>> A = scp.read_remote('irdata/subdir')
    """
    kwargs["remote"] = True
    if "merge" not in kwargs:
        kwargs["merge"] = False  # by default, no attempt to merge
    if "replace_existing" not in kwargs:
        kwargs["replace_existing"] = False  # by default we download only if needed.
    importer = Importer()
    return importer(file_or_dir, **kwargs)


def _write_downloaded_file(content, dst):
    if not dst.parent.exists():
        # create the eventually missing subdirectory
        dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(content)
    info_(f"{dst.name} has been downloaded and written in {dst.parent}")


def _get_url_content_and_save(url, dst, replace):

    if not replace and dst.exists():
        return

    try:
        r = requests.get(url, allow_redirects=True)

        # write downloaded file
        _write_downloaded_file(r.content, dst)

    except OSError:
        raise FileNotFoundError(f"Not found locally or on github (url:{url}")


def _download_full_testdata_directory():
    from spectrochempy.core import preferences as prefs

    datadir = prefs.datadir

    url = "https://github.com/spectrochempy/spectrochempy_data/archive/refs/heads/master.zip"

    resp = requests.get(url)
    zipfile = ZipFile(BytesIO(resp.content))
    files = [zipfile.open(file_name) for file_name in zipfile.namelist()]

    for file in files:
        name = file.name
        if name.endswith("/") or "testdata/" not in name:  # dir
            continue
        uncompressed = zipfile.read(name)
        p = list(pathclean(name).parts)[2:]
        dst = datadir.joinpath("/".join(p))
        _write_downloaded_file(uncompressed, dst)


def _download_from_url(url, dst, replace=False):
    # ##
    # ##    Do not forget to change the fork in the following url
    # ##
    if not str(url).startswith("https://"):
        # try on github
        url = (
            f"https://github.com/spectrochempy/spectrochempy_data/raw/master/"
            f"testdata/{url}"
        )
    # first determine if it is a directory
    r = requests.get(url + "/__index__", allow_redirects=True)
    index = None
    if r.status_code == 200:
        index = yaml.load(r.content, Loader=yaml.CLoader)

    if index is None:
        _get_url_content_and_save(url, dst, replace)

    else:
        for filename in index["files"]:
            _get_url_content_and_save(f"{url}/{filename}", dst / filename, replace)
        for folder in index["folders"]:
            _download_from_url(f"{url}/{folder}", dst / folder)


def _is_relative_to(path, base):
    # try to emulate the pathlib is_relative_to method which does not work on python
    # 3.7 (needed for Colab!)
    # TODO: replace if Colab is (unlikely) updated to a compatible version
    pparts = path.parts
    bparts = base.parts
    if bparts[-1] in pparts:
        idx = pparts.index(bparts[-1])
        pparts_base = pparts[: idx + 1]
        return pparts_base == bparts
    return False


def _relative_to(path, base):
    pparts = path.parts
    bparts = base.parts
    if bparts[-1] in pparts:
        idx = pparts.index(bparts[-1])
        return pathclean("/".join(pparts[idx + 1 :]))
    raise ValueError(
        f"'{path}' is not in the subpath of '{base}' OR one path is "
        f"relative and the other absolute."
    )


@_importer_method
def _read_remote(*args, **kwargs):
    from spectrochempy.core import preferences as prefs

    datadir = prefs.datadir

    dataset, path = args
    # path of the required files
    path = pathclean(path)

    if _is_relative_to(path, datadir):
        # try to make it relative for remote downloading
        relative_path = _relative_to(path, datadir)
    else:
        # assume it is already relative
        relative_path = path

    # in principle the data came from github. Try to download it

    replace = kwargs.pop("replace_existing", False)
    dst = datadir / relative_path
    if dst.name != "testdata":
        _download_from_url(relative_path, dst, replace)
    else:
        # we are going to download the whole testdata directory -> use a faster method
        _download_full_testdata_directory()

    if not kwargs.pop("download_only", False):
        read_method = kwargs.pop("read_method", read)
        return read_method(dataset, dst, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================


@_importer_method
def _read_dir(*args, **kwargs):
    _, directory = args
    directory = get_directory_name(directory)
    files = get_filenames(directory, **kwargs)
    datasets = []
    valid_extensions = list(zip(*FILETYPES))[0] + list(zip(*ALIAS))[0]
    for key in [key for key in files.keys() if key[1:] in valid_extensions]:
        if key:
            importer = Importer()
            nd = importer(files[key], **kwargs)
            if not isinstance(nd, list):
                nd = [nd]
            datasets.extend(nd)
    return datasets


@_importer_method
def _read_scp(*args, **kwargs):
    dataset, filename = args
    return dataset.load(filename, **kwargs)


@_importer_method
def _read_(*args, **kwargs):
    dataset, filename = args

    if kwargs.pop("remote", False):
        return Importer._read_remote(*args, **kwargs)
    elif not filename or filename.is_dir():
        return Importer._read_dir(*args, **kwargs)

    # protocol = kwargs.get("protocol", None)
    # if protocol and ".scp" in protocol:
    #     return dataset.load(filename, **kwargs)
    #
    # elif filename and filename.name in ("fid", "ser", "1r", "2rr", "3rrr"):
    #     # probably an Topspin NMR file
    #     return dataset.read_topspin(filename, **kwargs)
    # elif filename:
    #     # try scp format
    #     try:
    #         return dataset.load(filename, **kwargs)
    #     except Exception:
    #         # lets try some common format
    #         for key in ["omnic", "opus", "topspin", "labspec", "matlab", "jdx"]:
    #             try:
    #                 _read = getattr(dataset, f"read_{key}")
    #                 f = f"{filename}.{key}"
    #                 return _read(f, **kwargs)
    #             except Exception:
    #                 pass
    #         raise NotImplementedError


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
