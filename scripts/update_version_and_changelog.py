import sys
import re
from pathlib import Path
from datetime import date

from spectrochempy.utils import sh
from spectrochempy.utils.citation import Citation, Zenodo

SCRIPTS = Path(__file__).parent
PROJECT = SCRIPTS.parent
REFERENCE = PROJECT / "docs" / "userguide" / "reference"

CHANGELOG = PROJECT / "CHANGELOG.md"
CHANGELOGRST = REFERENCE / "changelog.rst"
CITATION = PROJECT / "CITATION.cff"
ZENODO = PROJECT / ".zenodo.json"


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
