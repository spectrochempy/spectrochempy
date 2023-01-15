import json
import sys
from datetime import date
from pathlib import Path

import yaml
from cffconvert.cli.create_citation import create_citation
from setuptools_scm import get_version

CI = Path(__file__).parent
PROJECT = CI.parent
CITATION = PROJECT / "CITATION.cff"
ZENODO = PROJECT / ".zenodo.json"
DOCS = PROJECT / "docs"
WN = DOCS / "whatsnew"

gitversion = get_version(root="..", relative_to=__file__)


class Zenodo:
    def __init__(self, infile=ZENODO):
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
            fid.write("\n")  # add a trailing blank line for pre-commit compat.

    def update_date(self):
        """
        Update the publication date metadata
        """
        self._js["publication_date"] = date.today().isoformat()

    def update_version(self, version=None):
        """
        Update the version string metadata
        """
        if version is None:
            version = gitversion
        self._js["version"] = ".".join(version.split(".")[:3])

    def __str__(self):
        return json.dumps(self._js, indent=2)


class Citation:
    def __init__(self, infile=CITATION):
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
        Update the released-date metadata .
        """
        self._citation.cffobj["date-released"] = date.today().isoformat()

    def update_version(self, version=None):
        """
        Update the version metadata.
        """
        if version is None:
            version = gitversion
        self._citation.cffobj["version"] = ".".join(version.split(".")[:3])


def make_citation(version):
    """"""
    citation = Citation()
    citation.load()
    citation.update_version(version)
    citation.update_date()
    print(citation)
    citation.save()


def make_zenodo(version):
    """"""
    zenodo = Zenodo()
    zenodo.load()
    zenodo.update_version(version)
    zenodo.update_date()
    print(zenodo)
    zenodo.save()


def make_release_note_index(revision):

    # remove old rev files
    files = WN.glob("v*.dev*.rst")
    for file in files:
        file.unlink()
    if (WN / "latest.rst").exists():
        (WN / "latest.rst").unlink()

    # Create or update file with the current version number
    if revision == "unreleased":
        revision = gitversion.split("+")[0]

    changelog_content = (WN / "changelog.rst").read_text()
    arr = changelog_content.split(".. _new_section")
    for i, item in enumerate(arr[:]):
        if item.strip().endswith("(do not delete this comment)"):
            # nothing has been added to this section, clear it totally
            arr[i] = ""
        else:
            arr[i] = item.strip() + "\n"

    changelog_content = "\n".join(arr)
    changelog_content = changelog_content.strip() + "\n"  # end of file

    changelog_content = changelog_content.replace("{{ revision }}", revision)

    if ".dev" in revision:
        (WN / "latest.rst").write_text(changelog_content)
    else:
        # in principle this happens for release, create the related rst file
        (WN / f"v{revision}.rst").write_text(changelog_content)
        # void changelog (keep only section titles)
        (WN / "changelog.rst").write_text(
            """What's new in revision {{ revision }}
---------------------------------------------------------------------------------------
.. do not remove the  `revision` marker. It will be replaced during doc building

These are the changes in SpectroChemPy-{{ revision }}. See :ref:`release` for a full changelog
including other versions of SpectroChemPy.

.. _new_section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)


.. _new_section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. _new_section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. _new_section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


 """
        )
    # Create the new index.rst file
    files = WN.glob("v*.rst")
    names = []
    for file in files:
        name = file.name
        names.append(name)
    names.sort()
    names.reverse()
    dicvers = {}
    for name in names:
        arr = name.split(".")
        base = ".".join(arr[:3])
        v = f"{arr[0][1]}.{arr[1]}"
        if v in dicvers:
            dicvers[v].append(base)
        else:
            dicvers[v] = [base]

    with open(WN / "index.rst", "w") as f:
        f.write(
            """.. _release:

*************
Release notes
*************

This is the list of changes to SpectroChemPy between each release. For full details,
see the `commit logs <https://github.com/spectrochempy/spectrochempy/commits/>`_.
For install and upgrade instructions, see :ref:`installation`.
"""
        )
        for i, vers in enumerate(dicvers):
            latest = "\n    latest" if i == 0 and ".dev" in revision else ""
            f.write(
                f"""
Version {vers}
--------------

.. toctree::
    :maxdepth: 2
{latest}
"""
            )
            li = sorted(dicvers[vers], key=lambda x: int(str.split(x, ".")[2]))
            li.reverse()
            for rev in li:
                f.write(f"    {rev}\n")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        new_revision = sys.argv[1]
    else:
        new_revision = "unreleased"

    if new_revision != "unreleased":
        make_citation(new_revision)
        make_zenodo(new_revision)
    make_release_note_index(new_revision)
