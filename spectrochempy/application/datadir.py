# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module provides the DataDir class.
"""
from os import environ
from pathlib import Path

import traitlets as tr

from spectrochempy.utils.file import find_or_create_spectrochempy_dir, pathclean


# ======================================================================================
# DataDir class
# ======================================================================================
class DataDir(tr.HasTraits):
    """
    A class used to determine the path to the testdata directory.
    """

    path = tr.Instance(Path)

    @tr.default("path")
    def _get_path_default(self, **kwargs):  # pragma: no cover
        super().__init__(**kwargs)

        # create a directory testdata in .spectrochempy to avoid an error
        # if the following do not work
        path = find_or_create_spectrochempy_dir() / "testdata"
        path.mkdir(exist_ok=True)

        # try to use the conda installed testdata (spectrochempy_data package)
        try:
            conda_env = environ["CONDA_PREFIX"]
            _path = Path(conda_env) / "share" / "spectrochempy_data" / "testdata"
            if not _path.exists():
                _path = (
                    Path(conda_env) / "share" / "spectrochempy_data"
                )  # depending on the version of spectrochempy_data
            if _path.exists():
                path = _path

        except KeyError:
            pass

        return path

    def listing(self):
        """
        Create a str representing a listing of the testdata folder.

        Returns
        -------
        `str`
            Display of the datadir content
        """
        strg = f"{self.path.name}\n"  # os.path.basename(self.path) + "\n"

        def _listdir(strg, initial, nst):
            nst += 1
            for fil in pathclean(initial).glob(
                "*"
            ):  # glob.glob(os.path.join(initial, '*')):
                filename = fil.name  # os.path.basename(f)
                if filename.startswith("."):  # pragma: no cover
                    continue
                if (
                    not filename.startswith("acqu")
                    and not filename.startswith("pulse")
                    and filename not in ["ser", "fid"]
                ):
                    strg += "   " * nst + f"|__{filename}\n"
                if fil.is_dir():
                    strg = _listdir(strg, fil, nst)
            return strg

        return _listdir(strg, self.path, -1)

    def __str__(self):
        return self.listing()

    def _repr_html_(self):  # pragma: no cover
        # _repr_html is needed to output in notebooks
        return self.listing().replace("\n", "<br/>").replace(" ", "&nbsp;")
