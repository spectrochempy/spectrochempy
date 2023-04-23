# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module define a generic class to import directories, files and contents.
"""
__all__ = ["read", "read_dir"]  # , "read_remote"]
__dataset_methods__ = __all__

import io
import re
from warnings import warn
from zipfile import ZipFile

import requests
import yaml
from traitlets import Dict, HasTraits, List, Type, Unicode

from spectrochempy.core import info_, warning_
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.exceptions import DimensionsCompatibilityError, ProtocolError
from spectrochempy.utils.file import (
    check_filename_to_open,
    get_directory_name,
    get_filenames,
    pathclean,
)

FILETYPES = [
    ("scp", "SpectroChemPy files (*.scp)"),
    ("omnic", "Nicolet OMNIC files and series (*.spa *.spg *.srs)"),
    ("soc", "Surface Optics Corp. (*.ddr *.hdr *.sdr)"),
    ("labspec", "LABSPEC exported files (*.txt)"),
    ("opus", "Bruker OPUS files (*.[0-9]*)"),
    (
        "topspin",
        "Bruker TOPSPIN fid or series or processed data files "
        "(fid ser 1[r|i] 2[r|i]* 3[r|i]*)",
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


# --------------------------------------------------------------------------------------
class Importer(HasTraits):
    # Private Importer class

    objtype = Type()
    datasets = List()
    files = Dict()
    default_key = Unicode()
    protocol = Unicode()

    protocols = Dict()
    filetypes = Dict()

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
            # if merge is not specified, but the args are provided as a single list,
            # then will are supposed to merge the datasets. If merge is specified then
            # it has priority.
            # This is not useful for the 1D datasets, as if they are compatible they
            # are merged automatically
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

            try:
                prefs = self.datasets[0].preferences
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
            and hasattr(args[0], "_implements")
            and args[0]._implements() in ["NDDataset"]
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

            read_ = getattr(self, f"_read_{key[1:]}")

            dataset = None
            try:
                # read locally or using url if filename is an url
                dataset = read_(self.objtype(), filename, **kwargs)

            except (FileNotFoundError, OSError) as exc:
                # file was not found.
                # it is an url we raise an error
                local_only = kwargs.get("local_only", False)
                if _is_url(filename) or local_only:
                    raise (FileNotFoundError) from exc

                # else, we try on github
                try:
                    # Try to get the file from github
                    kwargs["read_method"] = read_
                    info_(
                        "File/directory not found locally: Attempt to download it from "
                        "the GitHub repository `spectrochempy_data`..."
                    )
                    dataset = _read_remote(self.objtype(), filename, **kwargs)

                except (FileNotFoundError) as exc:
                    raise (FileNotFoundError) from exc

                except Exception as e:
                    warning_(str(e))

            except Exception as e:
                warning_(str(e))

            if dataset is not None:
                if not isinstance(dataset, list):
                    datasets.append(dataset)
                else:
                    datasets.extend(dataset)

        if len(datasets) > 1:
            datasets = self._do_merge(datasets, **kwargs)
            if kwargs.get("merge", False):
                datasets[0].name = pathclean(filename).stem
                datasets[0].filename = pathclean(filename)

        self.datasets.extend(datasets)

    def _do_merge(self, datasets, **kwargs):

        # several datasets returned (only if several files have been passed) and the `merge` keyword argument is False
        merged = kwargs.get("merge", False)
        shapes = {nd.shape if hasattr(nd, "shape") else None for nd in datasets}
        if len(shapes) == 1 and None not in shapes:
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


# --------------------------------------------------------------------------------------
# Public Generic Read function
# --------------------------------------------------------------------------------------

_docstring.get_sections(
    """
See Also
--------
read : Generic reader inferring protocol from the filename extension.
read_zip : Read Zip archives (containing spectrochempy readable files)
read_dir : Read an entire directory.
read_opus : Read OPUS spectra.
read_labspec : Read Raman LABSPEC spectra (:file:`.txt`\ ).
read_omnic : Read Omnic spectra (:file:`.spa`\ , :file:`.spg`\ , :file:`.srs`\ ).
read_soc : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`\ ).
read_galactic : Read Galactic files (:file:`.spc`\ ).
read_quadera : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
read_topspin : Read TopSpin Bruker NMR spectra.
read_csv : Read CSV files (:file:`.csv`\ ).
read_jcamp : Read Infrared JCAMP-DX files (:file:`.jdx`\ , :file:`.dx`\ ).
read_matlab : Read Matlab files (:file:`.mat`\ , :file:`.dso`\ ).
read_carroucell : Read files in a directory after a carroucell experiment.
""",
    sections=["See Also"],
    base="Importer",
)

_docstring.delete_params("Importer.see_also", "read")


@_docstring.dedent
def read(*paths, **kwargs):
    """
    Generic read method.

    This method is generally able to load experimental files based on extensions.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        * *e.g.,* ( filename1, filename2, ...,  \*\*kwargs )*

        If the list of filenames are enclosed into brackets:

        * *e.g.,* ( **[** *filename1, filename2, ...* **]**, \*\*kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.

        If a source is not provided (*i.e.,* no ``paths`` , nor ``content``\ ),
        a dialog box will be opened to select files.
    %(kwargs)s

    Returns
    -------
    object : `NDDataset` or list of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    protocol : `str`\ , optional
        ``Protocol`` used for reading. It can be one of {``'scp'``\ , ``'omnic'``\ ,
        ``'opus'``\ , ``'topspin'``\ , ``'matlab'``\ , ``'jcamp'``\ , ``'csv'``\ ,
        ``'excel'``\ }. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    merge : `bool`\ , optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged (stacked along the first
        dimension) is returned.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.
    description : `str`, optional
        A Custom description.
    origin : one of {``'omnic'``\ , ``'tga'``\ }, optional
        Used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra.
        Up to now only ``'omnic'`` and ``'tga'`` have been implemented.
    csv_delimiter : `str`\ , optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application to handle drag and drop of files into a
        Browser.
    iterdir : `bool`\ , optional, default: `True`
        If `True` and no filename was provided, all files present in the provided
        ``directory`` are returned (and merged if ``merge`` is `True`\ .
        It is assumed that all the files correspond to current reading protocol.

        .. versionchanged:: 0.6.2

            ``iterdir`` replace the deprecated ``listdir`` argument.

    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.

    See Also
    --------
    %(Importer.see_also.no_read)s

    Examples
    ---------
    Reading a single OPUS file  (providing a windows type filename relative
    to the default `~spectrochempy.preferences.datadir` )

    >>> scp.read('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative
    to the default ``datadir`` )
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> scp.NDDataset.read('irdata/OPUS/test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> scp.read(p)
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets).
    Note that a directory is specified

    >>> le = scp.read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword
    `merge`

    >>> scp.read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(le)
    3

    Read without a filename. This has the effect of opening a dialog for file(s)
    selection

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
        except TypeError as e:
            print(e)

    # deprecated kwargs
    listdir = kwargs.pop("listdir", True)
    if "listdir" in kwargs and "iterdir" not in kwargs:
        kwargs["iterdir"] = listdir
        warning_(
            "argument `listdir` is deprecated, use ìterdir` instead",
            category=DeprecationWarning,
        )

    return importer(*paths, **kwargs)


# for some reasons the doctring.getsection modify the signature of the function
# when used as a decorator, so we use it as a function
_docstring.get_sections(
    read.__doc__,
    sections=["Parameters", "Other Parameters", "Returns"],
    base="Importer",
)

_docstring.delete_params("Importer.see_also", "read_dir")


@_docstring.dedent
def read_dir(directory=None, **kwargs):
    """
    Read an entire directory.

    Open a list of readable files in a and store data/metadata in a dataset or a list of
    datasets according to the following rules :

    * 2D spectroscopic data (e.g. valid .spg files or matlab arrays, etc...) from
      distinct files are stored in distinct `NDdataset`\ s.
    * 1D spectroscopic data (e.g., :file:`.spa` files) in a given directory are merged
      into single `NDDataset`\ , providing their unique dimension are compatible.
      If not, an error is generated.
    * non-readable files are ignored

    Parameters
    ----------
    directory : str or pathlib
        Folder where are located the files to read.

    Returns
    --------
    %(Importer.returns)s
        Depending on the python version, the order of the datasets in the list
        may change.

    See Also
    --------
    %(Importer.see_also.no_read_dir)s

    Examples
    --------

    >>> scp.preferences.csv_delimiter = ','
    >>> A = scp.read_dir('irdata')
    >>> len(A)
    4

    >>> B = scp.NDDataset.read_dir()
    """
    kwargs["iterdir"] = True
    importer = Importer()
    return importer(directory, **kwargs)


# _docstring.delete_params("Importer.see_also", "read_remote")
# @_docstring.dedent
# def read_remote(file_or_dir, **kwargs):
#     """
#     Download and read files or an entire directory from any url
#
#     The first usage in spectrochempy is the loading of test files in the
#     `spectrochempy_data repository <https://github.com/spectrochempy/spectrochempy_data>`__\ .
#     This is done only if the data are not yet
#     downloaded and present in the `~spectrochempy.preferences.datadir` directory.
#
#     It can also be used to download and read file or directory from any url.
#
#     Parameters
#     ----------
#     path : `str`, `~pathlib.Path` object or an url.
#         When a file or folder is specified, it must be written as if it were present
#         locally exactly as for the `read` function. The correponding file or directory
#         is downloaded from the ``github spectrochemp_data`` repository.
#         Otherwise it should be a full and valid url.
#     %(kwargs)s
#
#     Returns
#     --------
#     %(Importer.returns)s
#
#     Other Parameters
#     ----------------
#     %(Importer.other_parameters)s
#
#     See Also
#     --------
#     %(Importer.see_also.no_read_remote)s
#
#     Examples
#     --------
#
#     >>> A = scp.read_remote('irdata/subdir')
#     """
#     kwargs["remote"] = True
#     importer = Importer()
#     return importer(file_or_dir, **kwargs)
#

# ======================================================================================
# Private read functions
# ======================================================================================
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
            if nd is not None:
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
    else:
        raise FileNotFoundError

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


# ======================================================================================
# Private functions
# ======================================================================================
def _is_url(filename):
    return (
        isinstance(filename, str)
        and re.match(r"http[s]?:[\/]{2}", filename) is not None
    )


def _openfid(filename, mode="rb", **kwargs):
    # Return a file ID

    # Check if Content has been passed?
    content = kwargs.get("content", False)

    # default encoding
    encoding = "utf-8"

    if _is_url(filename):
        # by default we set the read_only flag to True when reading remote url
        kwargs["read_only"] = kwargs.get("read_only", True)

        # use request to read the remote content
        r = requests.get(filename, allow_redirects=True)
        r.raise_for_status()
        content = r.content
        encoding = r.encoding

    else:
        # Transform filename to a Path object is not yet the case
        filename = pathclean(filename)

    # Create the file ID
    if content:
        # if a content has been passed, then it has priority
        fid = (
            io.BytesIO(content)
            if mode == "rb"
            else io.StringIO(content.decode(encoding))
        )
    else:
        fid = open(filename, mode=mode)

    return fid, kwargs


def _write_downloaded_file(content, dst):
    if not dst.parent.exists():
        # create the eventually missing subdirectory
        dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(content)
    info_(f"{dst.name} has been downloaded and written in {dst.parent}")


def _get_url_content_and_save(url, dst, replace, read_only=False):

    if not replace and dst.exists():
        return

    try:
        r = requests.get(url, allow_redirects=True)

        r.raise_for_status()

        # write downloaded file
        if not read_only:
            _write_downloaded_file(r.content, dst)

        # in all case return the content
        return r.content

    except OSError:
        raise FileNotFoundError(f"Not found locally or at url:{url}")


def _download_full_testdata_directory():
    from spectrochempy.core import preferences as prefs

    datadir = prefs.datadir

    url = "https://github.com/spectrochempy/spectrochempy_data/archive/refs/heads/master.zip"

    resp = requests.get(url)
    zipfile = ZipFile(io.BytesIO(resp.content))
    files = [zipfile.open(file_name) for file_name in zipfile.namelist()]

    for file in files:
        name = file.name
        if name.endswith("/") or "testdata/" not in name:  # dir
            continue
        uncompressed = zipfile.read(name)
        p = list(pathclean(name).parts)[2:]
        dst = datadir.joinpath("/".join(p))
        _write_downloaded_file(uncompressed, dst)


def _download_from_github(path, dst, replace=False):
    # download on github (always save the downloaded files)
    relative_path = str(pathclean(path).as_posix())
    path = (
        f"https://github.com/spectrochempy/spectrochempy_data/raw/master/"
        f"testdata/{relative_path}"
    )

    # first determine if it is a directory
    r = requests.get(path + "/__index__", allow_redirects=True)
    index = None
    if r.status_code == 200:
        index = yaml.load(r.content, Loader=yaml.CLoader)

    if index is None:
        return _get_url_content_and_save(path, dst, replace)

    else:
        # download folder
        for filename in index["files"]:
            _get_url_content_and_save(f"{path}/{filename}", dst / filename, replace)
        for folder in index["folders"]:
            _download_from_github(f"{relative_path}/{folder}", dst / folder)


def _is_relative_to(path, base):
    # try to emulate the pathlib is_relative_to method which does not work on python
    # 3.7 (needed for Colab!)
    # TODO: replace as Colab is updated to 3.9
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
    kwargs["merge"] = kwargs.get("merge", False)  # by default, no attempt to merge
    read_method = kwargs.pop("read_method", read)
    download_only = kwargs.pop("download_only", False)
    replace = kwargs.pop(
        "replace_existing", False
    )  # by default we download only if needed.

    # downloaded file
    # we try to download the github testdata
    path = pathclean(path)

    # we need to download additional files for topspin
    topspin = True if "topspin" in read_method.__name__ else False
    # we have to treat a special case: topspin, where the parent directory need
    # to be downloaded with the required file
    if topspin:
        savedpath = path
        m = re.match(r"(.*)(\/pdata\/\d+\/\d+[r|i]{1,2}|ser|fid)", str(path))
        if m is not None:
            path = pathclean(m[1])

    if _is_relative_to(path, datadir):
        # try to make it relative for remote downloading on github
        relative_path = _relative_to(path, datadir)
    else:
        # assume it is already relative
        relative_path = path

    # Try to download it
    dst = datadir / relative_path
    if dst.name == "testdata":
        # we are going to download the whole testdata directory
        # -> use a faster method
        _download_full_testdata_directory()
        return
    else:
        content = _download_from_github(relative_path, dst, replace)

    if not download_only:
        if content is None:
            if topspin:
                return read_method(
                    dataset, dst / _relative_to(savedpath, dst), **kwargs
                )
            else:
                return read_method(dataset, dst, **kwargs)
        else:
            return read_method(dataset, dst, content=content, **kwargs)
