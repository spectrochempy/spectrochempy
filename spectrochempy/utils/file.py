# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
File utilities.
"""
from os import environ
import re
import warnings
from pathlib import Path, WindowsPath, PosixPath

__all__ = [
    "get_filenames",
    "get_directory_name",
    "pathclean",
    "patterns",
    "check_filenames",
    "check_filename_to_open",
    "check_filename_to_save",
]


# ======================================================================================================================
# Utility functions
# ======================================================================================================================


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


def pathclean(paths):
    """
    Clean a path or a series of path in order to be compatible with windows and unix-based system.

    Parameters
    ----------
    paths :  str or a list of str
        Path to clean. It may contain windows or conventional python separators.

    Returns
    -------
    out : a pathlib object or a list of pathlib objets
        Cleaned path(s)

    Examples
    --------
    >>> from spectrochempy.utils import pathclean

    Using unix/mac way to write paths
    >>> filename = pathclean('irdata/nh4y-activation.spg')
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'

    or Windows
    >>> filename = pathclean("irdata\\\\nh4y-activation.spg")
    >>> filename.parent.name
    'irdata'

    Due to the escape character \\ in Unix, path string should be escaped \\\\ or the raw-string prefix `r` must be used
    as shown below
    >>> filename = pathclean(r"irdata\\nh4y-activation.spg")
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'
    """
    from spectrochempy.utils import is_windows

    def _clean(path):
        if isinstance(path, (Path, PosixPath, WindowsPath)):
            path = path.name
        if is_windows():
            path = WindowsPath(path)  # pragma: no cover
        else:  # some replacement so we can handle window style path on unix
            path = path.strip()
            path = path.replace("\\", "/")
            path = path.replace("\n", "/n")
            path = path.replace("\t", "/t")
            path = path.replace("\b", "/b")
            path = path.replace("\a", "/a")
            path = PosixPath(path)
        return Path(path)

    if paths is not None:
        if isinstance(paths, (str, Path, PosixPath, WindowsPath)):
            path = str(paths)
            return _clean(path).expanduser()
        elif isinstance(paths, (list, tuple)):
            return [_clean(p).expanduser() if isinstance(p, str) else p for p in paths]

    return paths


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
    ==========
    *args
        If passed it is a str, a list of str or a dictionary containing filenames or a byte's contents.
    **kwargs
        Optional keywords parameters. See Othe parameters

    Other Parameters
    ================
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
    ========
    check_filename_to_open
    check_filename_to_save

    Examples
    ========
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
                    "Two differents directory where specified (from args and keywords arg). "
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

    Returns
    --------
    out
        List of filenames.

    Examples
    --------
    """

    from spectrochempy.core import preferences as prefs
    from spectrochempy import NO_DIALOG

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

    from spectrochempy.core import preferences as prefs
    from spectrochempy import NO_DIALOG

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


# ..............................................................................
def check_filename_to_save(
    dataset, filename=None, save_as=True, confirm=True, **kwargs
):

    from spectrochempy import NO_DIALOG

    NODIAL = NO_DIALOG or "DOC_BUILDING" in environ

    if not filename or save_as:

        # no filename provided
        if filename is None or (NODIAL and pathclean(filename).is_dir()):
            filename = dataset.name
            filename = filename + kwargs.get("suffix", ".scp")

        if not NODIAL and confirm:

            from spectrochempy.core import save_dialog

            filename = save_dialog(
                caption=kwargs.pop("caption", "Save as ..."),
                filename=filename,
                filters=kwargs.pop("filetypes", ["All file types (*.*)"]),
                **kwargs,
            )
            if filename is None:
                # this is probably due to a cancel action for an open dialog.
                return

    if pathclean(filename).parent.resolve() == Path.cwd():
        return Path.cwd() / filename

    return pathclean(filename)


# ..........................................................................
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


# EOF
