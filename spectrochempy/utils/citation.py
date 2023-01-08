# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import json
import pathlib
import sys
from datetime import date

import yaml
from cffconvert.cli.create_citation import create_citation

__all__ = ["Citation", "Zenodo"]

sys.tracebacklimit = 2

HOME = pathlib.Path(__file__).parent.parent.parent


class Zenodo:
    def __init__(self, infile=HOME / ".zenodo.json"):
        self._infile = infile
        self._js = None

    def load(self):
        """
        Load the .zenodo.json file
        """
        with self._infile.open("r") as fid:
            self._js = json.load(fid)

    def save(self):
        """
        Write the .zenodo.json file
        """
        with self._infile.open("w") as fid:
            json.dump(self._js, fid, indent=2)

    def update_date(self):
        """
        Update the publication date metadata
        """
        self._js["publication_date"] = date.today().isoformat()

    def update_version(self, version=None):
        """
        Update the version string metadata
        """
        from spectrochempy.core import version as scpversion

        if version is None:
            version = scpversion
        self._js["version"] = ".".join(version.split(".")[:3])

    def __str__(self):
        return json.dumps(self._js, indent=2)


class Citation:
    def __init__(self, infile=HOME / "CITATION.cff"):
        self._infile = infile
        self._citation = None

    def __getattr__(self, key):
        if not self._citation:
            self.load()
        self._outputformat = {
            "apa": self._citation.as_apalike,
            "bibtex": self._citation.as_bibtex,
            "cff": self._citation.as_cff,
            "endnote": self._citation.as_endnote,
            "ris": self._citation.as_ris,
        }
        if key in self._outputformat.keys():
            return self.format(key)
        raise AttributeError(f"`{key}` attribute not found in the `Citation` object.")

    def load(self):
        """
        Load the CITATION.cff file
        """
        self._citation = create_citation(self._infile, url=None)
        try:
            self._citation.validate()
        except Exception as e:
            raise ImportError(e)

    def save(self):
        """
        Write the CITATION.cff file
        """
        with self._infile.open("w") as fid:
            fid.write(yaml.dump(self._citation.cffobj, indent=2))

    def __str__(self):
        return self.apa

    def format(self, fmt="apa"):
        """
        Return a str with citation in the given format

        Parameters
        ----------
        fmt : str, optional, default: "apa"
            Output format: "apa', "bibtex", "cff", "endnote" or "ris"

        Return
        ------
        str
            The citation in the given format

        Examples
        --------

        >>> citation = Citation()
        >>> apa = citation.format("apa")

        It is also possible to directly get the desired format using the name of the
        attribute:
        e.g.,

        >>> apa = citation.apa

        By default, printing, citation result is done with the apa format:

        >>> print(citation)
        Travert A., Fernandez C. ...
        """
        return self._outputformat[fmt]()

    def update_date(self):
        """
        Update the realesed-date metadata .
        """
        self._citation.cffobj["date-released"] = date.today().isoformat()

    def update_version(self, version=None):
        """
        Update the version metadata.
        """
        from spectrochempy.core import version as scpversion

        if version is None:
            version = scpversion
        self._citation.cffobj["version"] = ".".join(version.split(".")[:3])
