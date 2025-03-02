"""
Version and Release Notes Management Script.

This script manages project versioning and documentation by:
1. Updating version information in various project files
2. Managing citation information
3. Generating and updating release notes
4. Maintaining changelog documentation

The script handles these files:
- /data/CITATION.cff: Citation information
- /data/zenodo.json: Zenodo metadata
- /docs/whatsnew/: Release notes and changelog

Usage:
    python update_version_and_release_notes.py [version]
    If version is not provided, uses 'unreleased'
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import yaml

try:
    NO_CFFCONVERT = False
    from cffconvert.cli.create_citation import create_citation
except ImportError:
    NO_CFFCONVERT = True

from setuptools_scm import get_version

# Path configurations
WORKFLOWS = Path(__file__).parent.parent
PROJECT = WORKFLOWS.parent.parent
CITATION = PROJECT / "CITATION.cff"
ZENODO = PROJECT / "zenodo.json"
DOCS = PROJECT / "docs"
WN = DOCS / "sources" / "whatsnew"

gitversion = get_version(root=PROJECT, relative_to=__file__)


class Zenodo:
    """
    Handle Zenodo metadata file operations.

    This class manages the zenodo.json file which contains metadata for Zenodo
    repositories including version information and publication dates.
    """

    def __init__(self, infile=ZENODO):
        self._infile = infile
        self._js = None

    def load(self):
        """Load the zenodo.json file."""
        with self._infile.open("r") as fid:
            self._js = json.load(fid)

    def save(self):
        """Write the zenodo.json file."""
        with self._infile.open("w") as fid:
            json.dump(self._js, fid, indent=2)
            fid.write("\n")  # add a trailing blank line for pre-commit compat.

    def update_date(self):
        """Update the publication date metadata."""
        self._js["publication_date"] = date.today().isoformat()

    def update_version(self, version=None):
        """Update the version string metadata."""
        if version is None:
            version = gitversion
        self._js["version"] = ".".join(version.split(".")[:3])

    def __str__(self):
        return json.dumps(self._js, indent=2)


class Citation:
    """
    Manages citation information for the project.

    This class handles the CITATION.cff file, providing multiple citation formats
    and maintaining version/date information.
    """

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
        if key in self._outputformat:
            return self.format(key)
        raise AttributeError

    def load(self):
        """Load the CITATION.cff file."""
        self._citation = create_citation(self._infile, url=None)
        try:
            self._citation.validate()
        except Exception as e:
            raise ImportError(e) from None

    def save(self):
        """Write the CITATION.cff file."""
        with self._infile.open("w") as fid:
            fid.write(yaml.dump(self._citation.cffobj, indent=2))

    def __str__(self):
        return self.apa

    def format(self, fmt="apa"):
        """
        Return a str with citation in the given format.

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
        """Update the released-date metadata ."""
        self._citation.cffobj["date-released"] = date.today().isoformat()

    def update_version(self, version=None):
        """Update the version metadata."""
        if version is None:
            version = gitversion
        self._citation.cffobj["version"] = ".".join(version.split(".")[:3])


def make_citation(version):
    """
    Create or update the CITATION.cff file.

    Parameters
    ----------
    version : str
        Version string to use in citation

    """
    citation = Citation()
    citation.load()
    citation.update_version(version)
    citation.update_date()
    print(citation)  # noqa: T201
    citation.save()


def make_zenodo(version):
    """
    Create or update the zenodo.json file.

    Parameters
    ----------
    version : str
        Version string to use in Zenodo metadata

    """
    zenodo = Zenodo()
    zenodo.load()
    zenodo.update_version(version)
    zenodo.update_date()
    print(zenodo)  # noqa: T201
    zenodo.save()


def make_release_note_index(revision):
    """
    Generate and update release notes documentation.

    Parameters
    ----------
    revision : str
        Version string ('unreleased' or specific version)

    Notes
    -----
    This function:
    1. Cleans up old development version files
    2. Updates the changelog
    3. Creates version-specific release notes
    4. Generates an index of all release notes

    """
    # Clean up old dev files
    files = WN.glob("v*.dev*.rst")
    for file in files:
        file.unlink()
    if (WN / "latest.rst").exists():
        (WN / "latest.rst").unlink()

    # Handle version string
    if revision == "unreleased":
        revision = gitversion.split(".dev")[0]
        revision = revision + ".dev"

    # Process changelog content
    content = (WN / "changelog.rst").read_text()
    sections = re.split(r"^\.\. section$", content, flags=re.M)

    # Clean and organize sections
    header = re.sub(r"(\.\.\n(.*\n)*)", "", sections[0], count=0, flags=0)
    header = header.strip() + "\n"
    cleaned_sections = [header]

    for section in sections[1:]:
        if section.strip().endswith("(do not delete this comment)"):
            continue
        content = re.sub(
            r"(\.\. Add.*\(do not delete this comment\)\n)",
            "",
            section,
            count=0,
            flags=0,
        )
        content = content.strip() + "\n"
        cleaned_sections.append(content)

    # Generate final changelog
    changelog_content = "\n".join(cleaned_sections)
    changelog_content = changelog_content.replace("{{ revision }}", revision)

    # Write appropriate files based on version type
    if ".dev" in revision:
        (WN / "latest.rst").write_text(changelog_content)
    else:
        # Handle release version
        (WN / f"v{revision}.rst").write_text(changelog_content)
        (WN / "latest.rst").write_text(changelog_content)
        # Reset changelog template
        (WN / "changelog.rst").write_text(_get_changelog_template())

    # Generate index file
    _generate_release_index(revision)


def _get_changelog_template():
    """Return the template for a new changelog file."""
    return """
:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)


.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Dependency updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
"""


def _generate_release_index(revision):
    """
    Generate the release notes index file.

    Parameters
    ----------
    revision : str
        Current version being processed

    """
    # Collect and sort version files
    files = WN.glob("v*.rst")
    names = sorted([f.name for f in files], reverse=True)

    # Organize versions
    dicvers = {}
    for name in names:
        arr = name.split(".")
        base = ".".join(arr[:3])
        v = f"{arr[0][1]}.{arr[1]}"
        if v in dicvers:
            dicvers[v].append(base)
        else:
            dicvers[v] = [base]

    # Generate index content
    with open(WN / "index.rst", "w") as f:
        f.write(_get_index_header())

        for i, vers in enumerate(dicvers):
            latest = "\n    latest" if i == 0 and ".dev" in revision else ""
            f.write(
                f"""
Version {vers}
--------------

.. toctree::
    :maxdepth: 1
{latest}
""",
            )
            # Sort and write version entries
            li = sorted(dicvers[vers], key=lambda x: int(str.split(x, ".")[2]))
            li.reverse()
            for rev in li:
                f.write(f"    {rev}\n")


def _get_index_header():
    """Return the standard header for the release notes index."""
    return """.. _release:

*************
Release notes
*************

..
   Do not modify this file as it is automatically generated.
   See '.github/workflows/scripts/update_version_and_release_notes.py' if you need to change the output.

This is the list of changes to `SpectroChemPy` between each release. For full details,
see the `commit logs <https://github.com/spectrochempy/spectrochempy/commits/>`_ .
For install and upgrade instructions, see :ref:`installation`.
"""


if __name__ == "__main__":
    # Get version from command line or use 'unreleased'
    new_revision = sys.argv[1] if len(sys.argv) > 1 else "unreleased"

    # Update citation and Zenodo info for actual releases
    if new_revision != "unreleased" and not NO_CFFCONVERT:
        make_citation(new_revision)
        make_zenodo(new_revision)

    # Always update release notes
    make_release_note_index(new_revision)
