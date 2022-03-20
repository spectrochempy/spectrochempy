import sys
import re
from pathlib import Path
from datetime import date
import json
import yaml
from subprocess import run, PIPE, STDOUT
from setuptools_scm import get_version
from cffconvert.cli.create_citation import create_citation

SCRIPTS = Path(__file__).parent
PROJECT = SCRIPTS.parent
REFERENCE = PROJECT / "docs" / "userguide" / "reference"

CHANGELOG = PROJECT / "CHANGELOG.md"
CHANGELOGRST = REFERENCE / "changelog.rst"
CITATION = PROJECT / "CITATION.cff"
ZENODO = PROJECT / ".zenodo.json"

gitversion = get_version(root="..", relative_to=__file__)


class _ExecCommand:
    """
    Parameters
    ----------
    command : shell command to execute
    """

    def __init__(self, command):

        self.commands = [command]

    def __call__(self, *args, **kwargs):

        self.commands.extend(args)

        silent = kwargs.pop("silent", False)
        proc = run(
            self.commands, text=True, stdout=PIPE, stderr=STDOUT
        )  # capture_output=True)

        # TODO: handle error codes
        if not silent and proc.stdout:
            print(proc.stdout)
        return proc.stdout


class sh(object):
    """
    Utility to run subprocess run command as if they were functions.
    """

    def __getattr__(self, command):

        return _ExecCommand(command)

    def __call__(self, script, silent=False):
        # use to run shell script

        proc = run(script, text=True, shell=True, stdout=PIPE, stderr=STDOUT)

        if not silent:
            print(proc.stdout)
        return proc.stdout


sh = sh()


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
        Update the realesed-date metadata .
        """
        self._citation.cffobj["date-released"] = date.today().isoformat()

    def update_version(self, version=None):
        """
        Update the version metadata.
        """
        if version is None:
            version = gitversion
        self._citation.cffobj["version"] = ".".join(version.split(".")[:3])


def make_changelog(version):

    # check that a section named unreleased is present

    file = CHANGELOG
    md = file.read_text()

    if version != "unreleased":
        print(f'\n{"-" * 80}\nMake `changelogs`\n{"-" * 80}')

        vers = version.split(".")[:3]
        revision = ".".join(vers[:3])

        # split in sections
        lmd = re.split("\n##\s", md)
        lmd[1] = lmd[1].replace(
            "Unreleased\n", f"Version {revision} " f"[{date.today().isoformat()}]\n"
        )

        # rebuild the md file
        md = "\n## ".join(lmd)
    else:
        md = md.replace(
            "# What's new", "# What's new\n\n## Unreleased\n\n### NEW FEATURES\n*"
        )
    # file.write_text(md)

    sh(f"pandoc {CHANGELOG} -f  markdown -t rst -o {CHANGELOGRST}")
    print(f"`Complete what's new` log written to:\n{CHANGELOGRST}\n")


def make_citation(version):
    """"""
    citation = Citation()
    citation.load()
    citation.update_version(version)
    citation.update_date()
    print(citation)
    # citation.save()


def make_zenodo(version):
    """"""
    zenodo = Zenodo()
    zenodo.load()
    zenodo.update_version(version)
    zenodo.update_date()
    print(zenodo)
    # zenodo.save()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        new_version = sys.argv[1]
        if new_version != "unreleased":
            make_citation(new_version)
            make_zenodo(new_version)
        make_changelog(new_version)
