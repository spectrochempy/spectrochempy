# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
File utilities.
"""
import base64
import datetime
import json
import os
import pathlib
import pickle
import re
import warnings
from collections.abc import Mapping
from os import environ
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
from numpy.lib.format import read_array

from spectrochempy.utils.paths import pathclean


def make_zipfile(file, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64 (useful if files are larger than 4 GiB)
    (adapted from numpy)

    Parameters
    ----------
    file :  file or str
        The file to be zipped.
    **kwargs
        Additional keyword parameters.
        They are passed to the zipfile.ZipFile constructor.

    Returns
    -------
    zipfile
    """
    import zipfile

    kwargs["allowZip64"] = True
    return zipfile.ZipFile(file, **kwargs)


class ScpFile(Mapping):  # lgtm[py/missing-equals]
    """
    ScpFile(fid).

    (largely inspired by ``NpzFile`` object in numpy).

    `ScpFile` is used to load files stored in ``.scp`` or ``.pscp``
    format.

    It assumes that files in the archive have a ``.npy`` extension in
    the case of the dataset's ``.scp`` file format) ,  ``.scp``  extension
    in the case of project's ``.pscp`` file format and finally ``pars.json``
    files which contains other information on the structure and  attributes of
    the saved objects. Other files are ignored.

    Parameters
    ----------
    fid : file or str
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    """

    def __init__(self, fid):

        _zip = make_zipfile(fid)

        self.files = _zip.namelist()
        self.zip = _zip

        if hasattr(fid, "close"):
            self.fid = fid
        else:
            self.fid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the file.
        """
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def __del__(self):
        try:
            self.close()
        except AttributeError as e:
            if str(e) == "'ScpFile' object has no attribute 'zip'":
                pass
        except Exception as e:
            raise e

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):

        member = False
        ext = None

        if key in self.files:
            member = True
            _, ext = os.path.splitext(key)

        if member and ext in [".npy"]:
            f = self.zip.open(key)
            return read_array(f, allow_pickle=True)

        elif member and ext in [".scp"]:
            from spectrochempy.core.dataset.nddataset import NDDataset

            # f = io.BytesIO(self.zip.read(key))
            content = self.zip.read(key)
            return NDDataset.load(key, content=content)

        elif member and ext in [".json"]:
            content = self.zip.read(key)
            return json.loads(content, object_hook=json_decoder)

        elif member:
            return self.zip.read(key)

        else:
            raise KeyError("%s is not a file in the archive or is not " "allowed" % key)

    def __contains__(self, key):
        return self.files.__contains__(key)


def _insensitive_case_glob(pattern):
    def either(c):
        return f"[{c.lower()}{c.upper()}]" if c.isalpha() else c

    return "".join(map(either, pattern))


def patterns(filetypes, allcase=True):
    regex = r"\*\.*\[*[0-9-]*\]*\w*\**"
    patterns = []
    if not isinstance(filetypes, (list, tuple)):
        filetypes = [filetypes]
    for ft in filetypes:
        m = re.finditer(regex, ft)
        patterns.extend([match.group(0) for match in m])
    if not allcase:
        return patterns
    else:
        return [_insensitive_case_glob(p) for p in patterns]


def _get_file_for_protocol(f, **kwargs):
    protocol = kwargs.get("protocol", None)
    if protocol is not None:
        if isinstance(protocol, str):
            if protocol in ["ALL"]:
                protocol = "*"
            if protocol in ["opus"]:
                protocol = "*.0*"
            protocol = [protocol]

        lst = []
        for p in protocol:
            lst.extend(list(f.parent.glob(f"{f.stem}.{p}")))
        if not lst:
            return None
        else:
            return f.parent / lst[0]


def check_filenames(*args, **kwargs):
    """
    Return a list or a dictionary of filenames.

    Parameters
    ----------
    *args
        If passed it is a str, a list of str or a dictionary containing filenames or a byte's contents.
    **kwargs
        Optional keywords parameters. See Other parameters

    Other Parameters
    ----------------
    filename :
    filetypes :
    content :
    protocol :
    processed :
    expno :
    procno :
    listdir :
    glob :

    See Also
    --------
    check_filename_to_open
    check_filename_to_save

    Examples
    --------
    """
    from spectrochempy.core import preferences as prefs

    datadir = pathclean(prefs.datadir)

    filenames = None

    if args:
        if isinstance(args[0], (str, Path, PosixPath, WindowsPath)):
            # one or several filenames are passed - make Path objects
            filenames = pathclean(args)
        elif isinstance(args[0], bytes):
            # in this case, one or several byte contents has been passed instead of filenames
            # as filename where not given we passed the 'unnamed' string
            # return a dictionary
            return {pathclean(f"no_name_{i}"): arg for i, arg in enumerate(args)}
        elif isinstance(args[0], list) and isinstance(
            args[0][0], (str, Path, PosixPath, WindowsPath)
        ):
            filenames = pathclean(args[0])
        elif isinstance(args[0], list) and isinstance(args[0][0], bytes):
            return {pathclean(f"no_name_{i}"): arg for i, arg in enumerate(args[0])}
        elif isinstance(args[0], dict):
            # return directly the dictionary
            return args[0]

    if not filenames:
        # look into keywords (only the case where a str or pathlib filename is given are accepted)
        filenames = kwargs.pop("filename", None)
        filenames = [pathclean(filenames)] if pathclean(filenames) is not None else None

    # Look for content in kwargs
    content = kwargs.pop("content", None)
    if content:
        if not filenames:
            filenames = [pathclean("no_name")]
        return {filenames[0]: content}

    if not filenames:
        # no filename specified open a dialog
        filetypes = kwargs.pop("filetypes", ["all files (*)"])
        directory = pathclean(kwargs.pop("directory", None))
        filenames = get_filenames(
            directory=directory, dictionary=True, filetypes=filetypes, **kwargs
        )
    if filenames and not isinstance(filenames, dict):
        filenames_ = []
        for filename in filenames:
            # in which directory ?
            directory = filename.parent

            if directory.resolve() == Path.cwd() or directory == Path("."):
                directory = ""
            kw_directory = pathclean(kwargs.get("directory", None))
            if directory and kw_directory and directory != kw_directory:
                # conflict we do not take into account the kw.
                warnings.warn(
                    "Two different directory where specified (from args and keywords arg). "
                    "Keyword `directory` will be ignored!"
                )
            elif not directory and kw_directory:
                filename = kw_directory / filename

            # check if the file exists here
            if not directory or str(directory).startswith("."):
                # search first in the current directory
                directory = Path.cwd()

            f = directory / filename

            fexist = f if f.exists() else _get_file_for_protocol(f, **kwargs)

            if fexist is None:
                f = datadir / filename
                fexist = f if f.exists() else _get_file_for_protocol(f, **kwargs)

            if fexist:
                filename = fexist

            # Particular case for topspin where filename can be provided as a directory only
            # use of expno and procno
            if filename.is_dir() and "topspin" in kwargs.get("protocol", []):
                filename = _topspin_check_filename(filename, **kwargs)

            if not isinstance(filename, list):
                filename = [filename]

            filenames_.extend(filename)

        filenames = filenames_

    return filenames


def _topspin_check_filename(filename, **kwargs):

    if kwargs.get("listdir", False) or kwargs.get("glob", None) is not None:
        # when we list topspin dataset we have to read directories, not directly files
        # we can retrieve them using glob patterns
        glob = kwargs.get("glob", None)
        if glob:
            files_ = list(filename.glob(glob))
        elif not kwargs.get("processed", False):
            files_ = list(filename.glob("**/ser"))
            files_.extend(list(filename.glob("**/fid")))
        else:
            files_ = list(filename.glob("**/1r"))
            files_.extend(list(filename.glob("**/2rr")))
            files_.extend(list(filename.glob("**/3rrr")))
    else:
        expno = kwargs.pop("expno", None)
        procno = kwargs.pop("procno", None)

        if expno is None:
            expnos = sorted(filename.glob("[0-9]*"))
            expno = expnos[0] if expnos else expno

        # read a fid or a ser
        if procno is None:
            f = filename / str(expno)
            files_ = [f / "ser"] if (f / "ser").exists() else [f / "fid"]

        else:
            # get the adsorption spectrum
            f = filename / str(expno) / "pdata" / str(procno)
            if (f / "3rrr").exists():
                files_ = [f / "3rrr"]
            elif (f / "2rr").exists():
                files_ = [f / "2rr"]
            else:
                files_ = [f / "1r"]

    # depending on the glob patterns too many files may have been selected : restriction to the valid subset
    filename = []
    for item in files_:
        if item.name in ["fid", "ser", "1r", "2rr", "3rrr"]:
            filename.append(item)

    return filename


def get_filenames(*filenames, **kwargs):
    """
    Return a list or dictionary of the filenames of existing files, filtered by extensions.

    Parameters
    ----------
    filenames : `str` or pathlib object, `tuple` or `list` of strings of pathlib object, optional.
        A filename or a list of filenames.
        If not provided, a dialog box is opened to select files in the current
        directory if no `directory` is specified).
    **kwargs
        Other optional keyword parameters. See Other Parameters.

    Returns
    --------
    out
        List of filenames.

    Other Parameters
    ----------------
    directory : `str` or pathlib object, optional.
        The directory where to look at. If not specified, read in
        current directory, or in the datadir if unsuccessful.
    filetypes : `list`, optional, default=['all files, '.*)'].
        File type filter.
    dictionary : `bool`, optional, default=True
        Whether a dictionary or a list should be returned.
    listdir : bool, default=False
        Read all file (possibly limited by `filetypes` in a given `directory`.
    recursive : bool, optional,  default=False.
        Read also subfolders.

    Warnings
    --------
    if several filenames are provided in the arguments, they must all reside in the same directory!

    Examples
    --------
    """

    from spectrochempy import NO_DIALOG
    from spectrochempy.core import preferences as prefs

    NODIAL = (
        NO_DIALOG or "DOC_BUILDING" in environ
    ) and "KEEP_DIALOGS" not in environ  # flag to suppress dialog when doc is built or during full testing

    # allowed filetypes
    # -----------------
    # alias filetypes and filters as both can be used
    filetypes = kwargs.get("filetypes", kwargs.get("filters", ["all files (*)"]))

    # filenames
    # ---------
    if len(filenames) == 1 and isinstance(filenames[0], (list, tuple)):
        filenames = filenames[0]

    filenames = pathclean(list(filenames))

    directory = None
    if len(filenames) == 1:
        # check if it is a directory
        f = get_directory_name(filenames[0])
        if f and f.is_dir():
            # this specify a directory not a filename
            directory = f
            filenames = None
            NODIAL = True
    # else:
    #    filenames = pathclean(list(filenames))

    # directory
    # ---------
    kw_dir = pathclean(kwargs.pop("directory", None))
    if directory is None:
        directory = kw_dir

    if directory is not None:
        if filenames:
            # prepend to the filename (incompatibility between filename and directory specification
            # will result to a error
            filenames = [directory / filename for filename in filenames]
        else:
            directory = get_directory_name(directory)

    # check the parent directory
    # all filenames must reside in the same directory
    if filenames:
        parents = set()
        for f in filenames:
            parents.add(f.parent)
        if len(parents) > 1:
            raise ValueError(
                "filenames provided have not the same parent directory. "
                "This is not accepted by the readfilename function."
            )

        # use get_directory_name to complete eventual missing part of the absolute path
        directory = get_directory_name(parents.pop())

        filenames = [filename.name for filename in filenames]

    # now proceed with the filenames
    if filenames:

        # look if all the filename exists either in the specified directory,
        # else in the current directory, and finally in the default preference data directory
        temp = []
        for i, filename in enumerate(filenames):
            if not (directory / filename).exists():
                # the filename provided doesn't exists in the working directory
                # try in the data directory
                directory = pathclean(prefs.datadir)
                if not (directory / filename).exists():
                    raise IOError(f"Can't find  this filename {filename}")
            temp.append(directory / filename)

        # now we have checked all the filename with their correct location
        filenames = temp

    else:
        # no filenames:
        # open a file dialog
        # except if a directory is specified or listdir is True.

        getdir = kwargs.get(
            "listdir",
            directory is not None or kwargs.get("protocol", None) == ["topspin"]
            # or kwargs.get("protocol", None) == ["carroucell"],
        )

        if not getdir:
            # we open a dialogue to select one or several files manually
            if not NODIAL:

                from spectrochempy.core import open_dialog

                filenames = open_dialog(
                    single=False, directory=directory, filters=filetypes, **kwargs
                )
                if not filenames:
                    # cancel
                    return None

            elif environ.get("TEST_FILE", None) is not None:
                # happen for testing
                filenames = [pathclean(environ.get("TEST_FILE"))]

        else:

            if not NODIAL:
                from spectrochempy.core import open_dialog

                directory = open_dialog(
                    directory=directory, filters="directory", **kwargs
                )
                if not directory:
                    # cancel
                    return None

            elif NODIAL and not directory:
                directory = get_directory_name(environ.get("TEST_FOLDER"))

            elif NODIAL and kwargs.get("protocol", None) == ["topspin"]:
                directory = get_directory_name(environ.get("TEST_NMR_FOLDER"))

            if directory is None:
                return None

            filenames = []

            if kwargs.get("protocol", None) != ["topspin"]:
                # automatic reading of the whole directory
                for pat in patterns(filetypes):
                    if kwargs.get("recursive", False):
                        pat = f"**/{pat}"
                    filenames.extend(list(directory.glob(pat)))
            else:
                # Topspin directory detection
                filenames = [directory]

            # on mac case insensitive OS this cause doubling the number of files.
            # Eliminates doublons:
            filenames = list(set(filenames))

            filenames = pathclean(filenames)

        if not filenames:
            # problem with reading?
            return None

    # now we have either a list of the selected files
    if isinstance(filenames, list):
        if not all(
            isinstance(elem, (Path, PosixPath, WindowsPath)) for elem in filenames
        ):
            raise IOError("one of the list elements is not a filename!")

    # or a single filename
    if isinstance(filenames, (str, Path, PosixPath, WindowsPath)):
        filenames = [filenames]

    filenames = pathclean(filenames)
    for filename in filenames[:]:
        if filename.name.endswith(".DS_Store"):
            # sometime present in the directory (MacOSX)
            filenames.remove(filename)

    dictionary = kwargs.get("dictionary", True)
    protocol = kwargs.get("protocol", None)
    if dictionary and protocol != ["topspin"]:
        # make and return a dictionary
        filenames_dict = {}
        for filename in filenames:
            if filename.is_dir() and not protocol == ["carroucell"]:
                continue
            extension = filename.suffix.lower()
            if not extension:
                if re.match(r"^fid$|^ser$|^[1-3][ri]*$", filename.name) is not None:
                    extension = ".topspin"
            elif extension[1:].isdigit():
                # probably an opus file
                extension = ".opus"
            if extension in filenames_dict.keys():
                filenames_dict[extension].append(filename)
            else:
                filenames_dict[extension] = [filename]
        return filenames_dict
    else:
        return filenames


def get_directory_name(directory, **kwargs):
    """
    Return a valid directory name.

    Parameters
    ----------
    directory : `str` or `pathlib.Path` object, optional.
        A directory name. If not provided, a dialog box is opened to select a directory.

    Returns
    --------
    out: `pathlib.Path` object
        valid directory name.
    """

    from spectrochempy import NO_DIALOG
    from spectrochempy.core import preferences as prefs

    data_dir = pathclean(prefs.datadir)
    working_dir = Path.cwd()

    directory = pathclean(directory)

    if directory:
        if directory.is_dir():
            # nothing else to do
            return directory

        elif (working_dir / directory).is_dir():
            # if no parent directory: look at current working dir
            return working_dir / directory

        elif (data_dir / directory).is_dir():
            return data_dir / directory

        else:
            # raise ValueError(f'"{dirname}" is not a valid directory')
            warnings.warn(f'"{directory}" is not a valid directory')
            return None

    else:
        # open a file dialog
        directory = data_dir
        if not NO_DIALOG:  # this is for allowing test to continue in the background
            from spectrochempy.core import open_dialog

            directory = open_dialog(
                single=False, directory=working_dir, filters="directory", **kwargs
            )

        return pathclean(directory)


def check_filename_to_save(
    dataset, filename=None, save_as=False, confirm=True, **kwargs
):

    from spectrochempy import NO_DIALOG
    from spectrochempy.core import info_

    NODIAL = (NO_DIALOG or "DOC_BUILDING" in environ) and "KEEP_DIALOGS" not in environ

    if filename and pathclean(filename).parent.resolve() == Path.cwd():
        filename = Path.cwd() / filename

    if not filename or save_as or filename.exists():

        from spectrochempy.core import save_dialog

        # no filename provided
        open_diag = True
        caption = "Save as ..."
        if filename is None or (NODIAL and pathclean(filename).is_dir()):
            filename = dataset.name
            filename = filename + kwargs.get("suffix", ".scp")

        # existing filename provided
        elif filename.exists():
            if confirm:
                caption = "File exists. Confirm overwrite"
            else:
                info_(f"A file {filename} was present and has been overwritten.")
                open_diag = False

        if not NODIAL and open_diag:

            filename = save_dialog(
                caption=kwargs.pop("caption", caption),
                filename=filename,
                filters=kwargs.pop("filetypes", ["All file types (*.*)"]),
                **kwargs,
            )
            if filename is None:
                # this is probably due to a cancel action for an open dialog.
                return

    return pathclean(filename)


def check_filename_to_open(*args, **kwargs):
    # Check the args and keywords arg to determine the correct filename

    filenames = check_filenames(*args, **kwargs)

    if filenames is None:  # not args and
        # this is probably due to a cancel action for an open dialog.
        return

    if not isinstance(filenames, dict):

        if len(filenames) == 1 and filenames[0] is None:
            raise (FileNotFoundError)

        # deal with some specific cases
        key = filenames[0].suffix.lower()
        if not key:
            if re.match(r"^fid$|^ser$|^[1-3][ri]*$", filenames[0].name) is not None:
                key = ".topspin"
        if key[1:].isdigit():
            # probably an opus file
            key = ".opus"
        return {key: filenames}

    elif len(args) > 0 and args[0] is not None:
        # args where passed so in this case we have directly byte contents instead of filenames only
        contents = filenames
        return {"frombytes": contents}

    else:
        # probably no args (which means that we are coming from a dialog or from a full list of a directory
        return filenames


def fromisoformat(s):
    try:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%Z")
    except Exception:
        date = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return date


# ======================================================================================
# JSON UTILITIES
# ======================================================================================


def json_decoder(dic):
    """
    Decode a serialised json object.
    """
    from spectrochempy.core.units import Quantity, Unit

    if "__class__" in dic:

        klass = dic["__class__"]
        if klass == "DATETIME":
            return fromisoformat(dic["isoformat"])
        elif klass == "DATETIME64":
            return np.datetime64(dic["isoformat"])
        elif klass == "NUMPY_ARRAY":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))
            elif "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"])
        elif klass == "PATH":
            return Path(dic["str"])
        elif klass == "QUANTITY":
            return Quantity.from_tuple(dic["tuple"])
        elif klass == "UNIT":
            return Unit(dic["str"])
        elif klass == "COMPLEX":
            if "base64" in dic:
                return pickle.loads(base64.b64decode(dic["base64"]))
            elif "tolist" in dic:
                return np.array(dic["tolist"], dtype=dic["dtype"]).data[()]

        raise TypeError(dic["__class__"])

    return dic


def json_serialiser(byte_obj, encoding=None):
    """
    Return a serialised json object.
    """
    from spectrochempy.core.dataset.mixins.ndplot import PreferencesSet
    from spectrochempy.core.units import Quantity, Unit

    if byte_obj is None:
        return None

    elif hasattr(byte_obj, "implements"):

        objnames = byte_obj.__dir__()

        # particular case of Linear Coordinates
        if byte_obj.implements("LinearCoord"):
            objnames.remove("data")

        dic = {}
        for name in objnames:

            if (
                name in ["readonly"]
                or (name == "dims" and "datasets" in objnames)
                or [name in ["parent", "name"] and isinstance(byte_obj, PreferencesSet)]
                and name not in ["created", "modified", "acquisition_date"]
            ):
                val = getattr(byte_obj, name)
            else:
                val = getattr(byte_obj, f"_{name}")

            # Warning with parent-> circular dependencies!
            if name != "parent":
                dic[name] = json_serialiser(val, encoding=encoding)
        return dic

    elif isinstance(byte_obj, (str, int, float, bool)):
        return byte_obj

    elif isinstance(byte_obj, np.bool_):
        return bool(byte_obj)

    elif isinstance(byte_obj, (np.float64, np.float32, float)):
        return float(byte_obj)

    elif isinstance(byte_obj, (np.int64, np.int32, int)):
        return int(byte_obj)

    elif isinstance(byte_obj, tuple):
        return tuple([json_serialiser(v, encoding=encoding) for v in byte_obj])

    elif isinstance(byte_obj, list):
        return [json_serialiser(v, encoding=encoding) for v in byte_obj]

    elif isinstance(byte_obj, dict):
        dic = {}
        for k, v in byte_obj.items():
            dic[k] = json_serialiser(v, encoding=encoding)
        return dic

    elif isinstance(byte_obj, datetime.datetime):
        return {
            "isoformat": byte_obj.strftime("%Y-%m-%dT%H:%M:%S.%f%Z"),
            "__class__": "DATETIME",
        }

    elif isinstance(byte_obj, np.datetime64):
        return {
            "isoformat": np.datetime_as_string(byte_obj, timezone="UTC"),
            "__class__": "DATETIME64",
        }

    elif isinstance(byte_obj, np.ndarray):
        if encoding is None:
            dtype = byte_obj.dtype
            if str(byte_obj.dtype).startswith("datetime64"):
                byte_obj = np.datetime_as_string(byte_obj, timezone="UTC")
            return {
                "tolist": json_serialiser(byte_obj.tolist(), encoding=encoding),
                "dtype": str(dtype),
                "__class__": "NUMPY_ARRAY",
            }
        else:
            return {
                "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": "NUMPY_ARRAY",
            }

    elif isinstance(byte_obj, pathlib.PosixPath):
        return {"str": str(byte_obj), "__class__": "PATH"}

    elif isinstance(byte_obj, Unit):
        strunits = f"{byte_obj:D}"
        return {"str": strunits, "__class__": "UNIT"}

    elif isinstance(byte_obj, Quantity):
        return {
            "tuple": json_serialiser(byte_obj.to_tuple(), encoding=encoding),
            "__class__": "QUANTITY",
        }

    elif isinstance(byte_obj, (np.complex128, np.complex64, np.complex)):
        if encoding is None:
            return {
                "tolist": json_serialiser(
                    [byte_obj.real, byte_obj.imag], encoding=encoding
                ),
                "dtype": str(byte_obj.dtype),
                "__class__": "COMPLEX",
            }
        else:
            return {
                "base64": base64.b64encode(pickle.dumps(byte_obj)).decode(),
                "__class__": "COMPLEX",
            }

    raise ValueError(f"No encoding handler for data type {type(byte_obj)}")


# EOF
