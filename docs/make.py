# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201
"""
Clean, build, and release the HTML documentation for SpectroChemPy.

.. code-block:: bash

    python make.py [options]


where optional parameters indicates which job(s) is(are) to perform.
"""

import argparse
import multiprocessing as mp
import shutil
import sys
import warnings
import zipfile
from os import environ
from pathlib import Path

from sphinx.application import Sphinx

try:
    from sphinx.deprecation import RemovedInSphinx70Warning

    warnings.filterwarnings(action="ignore", category=RemovedInSphinx70Warning)

except ImportError:
    from sphinx.deprecation import RemovedInSphinx10Warning

    warnings.filterwarnings(action="ignore", category=RemovedInSphinx10Warning)


from apigen import Apigen

from spectrochempy.api import preferences as prefs
from spectrochempy.api import version
from spectrochempy.utils.file import download_testdata
from spectrochempy.utils.system import sh

warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)

warnings.filterwarnings(action="ignore", module="debugpy")
warnings.filterwarnings(action="ignore", category=FutureWarning)

# CONSTANT
PROJECTNAME = "spectrochempy"
REPO_URI = f"spectrochempy/src/{PROJECTNAME}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "www.spectrochempy.fr"

# GENERAL PATHS
DOCS = Path(__file__).parent
TEMPLATES = DOCS / "_templates"
STATIC = DOCS / "_static"
PROJECT = DOCS.parent
DOCREPO = PROJECT / "build"
DOCTREES = DOCREPO / "~doctrees"
HTML = DOCREPO / "html"
DOWNLOADS = HTML / "downloads"
SOURCES = PROJECT / PROJECTNAME

# DOCUMENTATION SRC PATH
SRC = DOCS
USERGUIDE = SRC / "userguide"
GETTINGSTARTED = SRC / "gettingstarted"
DEVGUIDE = SRC / "devguide"
REFERENCE = SRC / "reference"

# generated by sphinx
API = REFERENCE / "generated"
DEV_API = DEVGUIDE / "generated"
GALLERY = GETTINGSTARTED / "examples" / "gallery"

# create the testdata directory

datadir = prefs.datadir
# this process is relatively long, so we do not want to do it several time:
download_testdata()  # (download done using mamba install spectrochempy_data)


# ======================================================================================
# Class BuildDocumentation
# ======================================================================================
class BuildDocumentation:
    def __init__(
        self,
        delnb=False,
        noapi=False,
        noexec=False,
        tutorials=False,
        warningiserror=False,
        verbosity=0,
        jobs="auto",
        whatsnew=False,
    ):
        # determine if we are in the development branch (latest) or master (stable)
        if "+" in version:
            self._doc_version = "dirty"
        elif "dev" in version:
            print("\n\nWe are creating the latest (dev) documentation.\n")
            self._doc_version = "latest"
        else:
            print("\n\nWe are creating the stable documentation.\n")
            self._doc_version = "stable"

        self.delnb = delnb
        self.noapi = noapi
        self.noexec = noexec
        self.tutorials = tutorials
        self.warningiserror = warningiserror
        self.verbosity = verbosity

        # Determine number of jobs
        if jobs == "auto":
            jobs = mp.cpu_count()
        else:
            try:
                jobs = int(jobs)
            except ValueError:
                print("Error: --jobs argument must be an integer or 'auto'")
                return 1
        self.jobs = jobs

        self.whatsnew = whatsnew

        if noapi and not whatsnew:  # API is included when whatsnew is compiled
            environ["SPHINX_NOAPI"] = "noapi"
            return None
        return None

    @staticmethod
    def _delnb():
        # Remove all ipynb before git commit
        for nb in SRC.rglob("**/*.ipynb"):
            sh.rm(nb)
        for nbch in SRC.glob("**/.ipynb_checkpoints"):
            sh(f"rm -r {nbch}")

    @staticmethod
    def _make_redirection_page():
        # create an index page at the site root to redirect to the latest version.

        html = """
        <html>
        <head>
        <title>Processing, analyzing and modeling spectroscopic data</title>
        <meta http-equiv="refresh" content="0; URL=latest">
        </head>
        <body>
            <h1>Processing, analyzing and modeling spectroscopic data</H1>
            <p>
                SpectroChemPy is a framework for processing, analyzing and modeling <strong>Spectro</strong>scopic
                data for <strong>Chem</strong>istry with <strong>Py</strong>thon. It is a cross-platform software, running on Linux, Windows or OS X.
            </p>
        </body>
        </html>
        """
        with open(HTML / "index.html", "w") as f:
            f.write(html)

    @staticmethod
    def _confirm(action):
        # private method to ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(
                f"OK to continue `{action}` Y[es]/[N[o] ? ",
            ).lower()
        return answer[:1] == "y"

    @staticmethod
    def _sync_notebooks():
        # Use  jupytext to sync py and ipynb files in userguide

        pyfiles = set()
        print(f"\n{'-' * 80}\nSync *.py and *.ipynb using jupytex\n{'-' * 80}")

        py = list(SRC.glob("**/*.py"))
        py.extend(list(SRC.glob("**/*.ipynb")))

        for f in py[:]:
            # do not consider some files
            if (
                "generated" in f.parts
                or ".ipynb_checkpoints" in f.parts
                or "gallery" in f.parts
                or "examples" in f.parts
                or "sphinxext" in f.parts
            ) or f.name in ["conf.py", "make.py", "apigen.py"]:
                continue
            # add only the full path without suffix
            pyfiles.add(f.with_suffix(""))

        count = 0
        for item in pyfiles:
            py = item.with_suffix(".py")
            ipynb = item.with_suffix(".ipynb")

            args = None

            if not ipynb.exists():
                args = [
                    "--update-metadata",
                    '{"jupytext": {"notebook_metadata_filter":"all"}}',
                    "--to",
                    "ipynb",
                    py,
                ]

            elif not py.exists():
                args = [
                    "--update-metadata",
                    '{"jupytext": {"notebook_metadata_filter":"all"}}',
                    "--to",
                    "py:percent",
                    ipynb,
                ]

            if args is not None:
                print(f"sync: {item}")
                count += 1

                sh.jupytext(*args, silent=False)

        if count == 0:
            print("\nAll notebooks are up-to-date and synchronised with py files")
        print("\n")

    def _make_dirs(self):
        # Create the directories required to build the documentation.

        doc_version = self._doc_version

        # Create regular directories.
        build_dirs = [
            DOCREPO,
            DOCTREES,
            HTML,
            DOCTREES / doc_version,
            HTML / doc_version,
            DOWNLOADS,
        ]
        for d in build_dirs:
            Path.mkdir(d, parents=True, exist_ok=True)

    @staticmethod
    def _apigen():
        print(f"\n{'-' * 80}\nRegenerate the reference API list\n{'-' * 80}")

        Apigen()

    # COMMANDS
    # ----------------------------------------------------------------------------------
    def _make_docs(self, builder="html"):
        # Make the html documentation

        doc_version = self._doc_version

        if builder not in [
            "html",
        ]:
            raise ValueError('Not a supported builder: Must be "html"')

        BUILDDIR = DOCREPO / builder

        print(
            f"{'#' * 80}\n"
            f"Building {builder.upper()} documentation ({doc_version.capitalize()} "
            f"version : {version})"
            f"\n in {BUILDDIR}"
            f"\n{'#' * 80}"
        )

        # self._make_dirs()

        # APIGEN?
        if not self.noapi:
            self._apigen()

        self._sync_notebooks()

        # run sphinx
        print(f"{'-' * 80}\n")
        print(f"\n{builder.upper()} BUILDING")
        print(f"{'-' * 80}\n")
        srcdir = confdir = DOCS
        outdir = f"{BUILDDIR}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"

        sp = Sphinx(
            str(srcdir),
            str(confdir),
            str(outdir),
            str(doctreesdir),
            builder,
            warningiserror=self.warningiserror,
            parallel=self.jobs,
            verbosity=self.verbosity,
        )
        if self.noexec:
            sp.config.nbsphinx_execute = "never"
            sp.config.plot_gallery = 0

        res = sp.build()

        print(
            f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
            f"are in {outdir}."
        )

        if doc_version == "stable":
            doc_version = "latest"
            # make also the latest identical
            sh(f"rm -rf {BUILDDIR}/latest")
            sh(f"cp -r  {BUILDDIR}/stable {BUILDDIR}/latest")

        if builder == "html":
            self._make_redirection_page()

        return res

    def clean(self):
        # Clean/remove the built documentation.

        print(f"\n{'-' * 80}\nCleaning\n{'-' * 80}")

        doc_version = self._doc_version

        shutil.rmtree(HTML / doc_version, ignore_errors=True)
        print(f"removed {HTML / doc_version}")
        shutil.rmtree(DOCTREES / doc_version, ignore_errors=True)
        print(f"removed {DOCTREES / doc_version}")
        shutil.rmtree(API, ignore_errors=True)
        print(f"removed {API}")
        shutil.rmtree(DEV_API, ignore_errors=True)
        print(f"removed {DEV_API}")
        shutil.rmtree(GALLERY, ignore_errors=True)
        print(f"removed {GALLERY}")

        self._delnb()

    def html(self):
        """Generate HTML documentation."""
        return self._make_docs("html")

    def tutorials(self):
        # make tutorials.zip
        print(f"\n{'-' * 80}\nMake tutorials.zip\n{'-' * 80}")

        # clean notebooks output
        for nb in DOCS.rglob("**/*.ipynb"):
            # This will erase all notebook output
            sh(
                f"jupyter nbconvert "
                f"--ClearOutputPreprocessor.enabled=True --inplace {nb}",
                silent=True,
            )

        # make zip of all ipynb
        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for inb in path.rglob("**/*.ipynb"):
                if ".ipynb_checkpoints" in inb.parent.suffix:
                    continue
                basename = inb.stem
                sh(
                    f"jupyter nbconvert {inb} --to notebook"
                    f" --ClearOutputPreprocessor.enabled=True"
                    f" --stdout > out_{basename}.ipynb"
                )
                sh(f"rm {inb}", silent=True)
                sh(f"mv out_{basename}.ipynb {inb}", silent=True)
                arcnb = str(inb).replace(str(path), str(dest))
                ziph.write(inb, arcname=arcnb)

        zipf = zipfile.ZipFile("~notebooks.zip", "w", zipfile.ZIP_STORED)
        zipdir(SRC, "notebooks", zipf)
        zipdir(GALLERY / "auto_examples", Path("notebooks") / "examples", zipf)
        zipf.close()

        sh(
            f"mv ~notebooks.zip "
            f"{DOWNLOADS}/{self.doc_version}-{PROJECTNAME}-notebooks.zip"
        )


# ======================================================================================
def main():
    environ["DOC_BUILDING"] = "yes"

    commands = [
        method for method in dir(BuildDocumentation) if not method.startswith("_")
    ]

    scommands = ",".join(commands)
    parser = argparse.ArgumentParser(
        description="Build documentation for SpectroChemPy",
        epilog=f"Commands: {scommands}",
    )

    parser.add_argument(
        "command", nargs="?", default="html", help=f"available commands: {scommands}"
    )

    parser.add_argument(
        "--del-nb", "-D", help="delete all ipynb files", action="store_true"
    )

    parser.add_argument(
        "--no-api",
        "-A",
        help="execute a full regeneration of the api",
        action="store_true",
    )

    parser.add_argument(
        "--no-exec",
        "-E",
        help="do not execute notebooks",
        action="store_true",
    )

    parser.add_argument(
        "--upload-tutorials", "-T", help="zip and upload tutorials", action="store_true"
    )

    parser.add_argument(
        "--warning-is-error",
        "-W",
        action="store_true",
        help="fail if warnings are raised",
    )

    parser.add_argument(
        "-v",
        action="count",
        dest="verbosity",
        default=0,
        help=(
            "increase verbosity (can be repeated), passed to the sphinx build command"
        ),
    )

    parser.add_argument(
        "--single",
        metavar="FILENAME",
        type=str,
        default=None,
        help=(
            "filename (relative to the 'source' folder) of section or method name to "
            "compile, e.g. "
            "userguide/analysis.rst"
            ", 'spectrochempy.EFA'"
        ),
    )

    parser.add_argument(
        "--whatsnew",
        default=False,
        help="only build whatsnew (and api for links)",
        action="store_true",
    )

    parser.add_argument(
        "--jobs", "-j", default="auto", help="number of jobs used by sphinx-build"
    )

    args = parser.parse_args()

    if args.command not in commands:
        parser.print_help(sys.stderr)
        print(f"Unknown command {args.command}.")
        return 1

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("by default, command is set to html")
        args.html = True

    build = BuildDocumentation(
        delnb=args.del_nb,
        noapi=args.no_api,
        noexec=args.no_exec,
        tutorials=args.upload_tutorials,
        warningiserror=args.warning_is_error,
        verbosity=args.verbosity,
        jobs=args.jobs,
        whatsnew=args.whatsnew,
    )

    buildcommand = getattr(build, args.command)
    res = buildcommand()
    if res is None:
        res = 0
    return res

    del environ["DOC_BUILDING"]
    return None


# ======================================================================================
if __name__ == "__main__":
    sys.exit(main())
