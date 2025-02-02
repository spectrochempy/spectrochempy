# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201,S603
"""
Clean, build, and release the HTML documentation for SpectroChemPy.

.. code-block:: bash

    python make.py [options]


where optional parameters indicates which job(s) is(are) to perform.
"""

import argparse
import multiprocessing as mp
import shlex
import shutil
import sys
import warnings
import zipfile
from contextlib import suppress
from os import environ
from pathlib import Path
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import run

from sphinx.application import Sphinx

# Suppress specific warnings from Sphinx
with suppress(Exception):
    from sphinx.deprecation import RemovedInSphinx80Warning

    warnings.filterwarnings(action="ignore", category=RemovedInSphinx80Warning)

with suppress(Exception):
    from sphinx.deprecation import RemovedInSphinx90Warning

    warnings.filterwarnings(action="ignore", category=RemovedInSphinx90Warning)

with suppress(Exception):
    from sphinx.deprecation import RemovedInSphinx10Warning

    warnings.filterwarnings(action="ignore", category=RemovedInSphinx10Warning)

# Suppress other specific warnings
warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)

warnings.filterwarnings(action="ignore", module="debugpy")
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    action="ignore",
    message="Gallery in version 0.18.0 is not supported by Jupytext",
    category=UserWarning,
)

# CONSTANT
PROJECTNAME = "spectrochempy"
REPO_URI = f"spectrochempy/src/{PROJECTNAME}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "www.spectrochempy.fr"

# GENERAL PATHS
DOCS = Path(__file__).parent
TEMPLATES = DOCS / "templates"
STATIC = DOCS / "static"
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


# ======================================================================================
# Class BuildDocumentation
# ======================================================================================
class BuildDocumentation:
    """A class to build and manage the documentation for SpectroChemPy."""

    def __init__(self, **kwargs):
        self.settings = self._init_settings(kwargs)
        self._doc_version = None
        self._version = None

    def _determine_version(self):
        """Determine the version of the documentation to build."""
        from spectrochempy.api import version  # Import here to avoid unnecessary delay

        if "+" in version:
            return version, "dirty"
        return version, "latest" if "dev" in version else "stable"

    def _init_settings(self, kwargs):
        """Initialize settings from keyword arguments."""
        settings = {
            "delnb": kwargs.get("delnb", False),
            "noapi": kwargs.get("noapi", False),
            "noexec": kwargs.get("noexec", False),
            "tutorials": kwargs.get("tutorials", False),
            "warningiserror": kwargs.get("warningiserror", False),
            "verbosity": kwargs.get("verbosity", 0),
            "jobs": self._get_jobs(kwargs.get("jobs", "auto")),
            "whatsnew": kwargs.get("whatsnew", False),
        }

        if settings["noapi"] and not settings["whatsnew"]:
            environ["SPHINX_NOAPI"] = "noapi"

        return settings

    def _get_jobs(self, jobs):
        """Get the number of jobs to use for building the documentation."""
        if jobs == "auto":
            return mp.cpu_count()
        try:
            return int(jobs)
        except ValueError:
            print("Error: --jobs argument must be an integer or 'auto'")
            return 1

    @staticmethod
    def _delnb():
        """Remove all Jupyter notebook files."""
        for nb in SRC.rglob("**/*.ipynb"):
            sh.rm(nb)
        for nbch in SRC.glob("**/.ipynb_checkpoints"):
            sh(f"rm -r {nbch}")

    @staticmethod
    def _confirm(action):
        """Ask user to confirm an action by entering Y or N (case-insensitive)."""
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(
                f"OK to continue `{action}` Y[es]/[N[o] ? ",
            ).lower()
        return answer[:1] == "y"

    def _sync_notebooks(self):
        """Improved notebook synchronization with better error handling."""
        for item in self._get_notebook_files():
            try:
                self._sync_notebook_pair(item)
            except Exception as e:
                print(f"Failed to sync {item}: {e}")

    def _get_notebook_files(self):
        """Get a set of notebook files to be synchronized."""
        pyfiles = set()
        print(f"\n{'-' * 80}\nSync *.py and *.ipynb using jupytext\n{'-' * 80}")

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

        return pyfiles

    def _sync_notebook_pair(self, item):
        """Synchronize a pair of notebook and script files."""
        py = item.with_suffix(".py")
        ipynb = item.with_suffix(".ipynb")

        # Determine the file to use for setting up pairing
        file_to_pair = py if py.exists() else ipynb

        # Use jupytext --sync to synchronize the notebook/script pair
        try:
            result = sh(
                f"jupytext --sync {file_to_pair}",
                silent=True,
            )
            if "Updating" in result:
                updated_files = [
                    line.split()[-1]
                    for line in result.splitlines()
                    if "Updating" in line
                ]
                for updated_file in updated_files:
                    print(f"Updated: {updated_file}")
            else:
                print(f"Unchanged: {file_to_pair}")
        except Exception as e:
            print(f"Warning: Failed to synchronize {item}: {str(e)}")

    def _make_dirs(self):
        """Create the directories required to build the documentation."""
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
        """Regenerate the reference API list."""
        from apigen import Apigen  # Import here to avoid unnecessary delay

        print(f"\n{'-' * 80}\nRegenerate the reference API list\n{'-' * 80}")

        Apigen()

    def _get_previous_versions(self):
        """Get a list of previous versions from the HTML directory."""
        versions = []
        for item in HTML.iterdir():
            if item.is_dir() and item.name not in ["stable", "downloads"]:
                versions.append(item.name)
        return versions

    # COMMANDS
    # ----------------------------------------------------------------------------------
    def _make_docs(self, builder="html"):
        """Simplified documentation building process."""
        if builder != "html":
            raise ValueError("Only HTML builder is supported")

        self._prepare_build()
        build_result = self._run_sphinx_build(builder)
        self._post_build(builder)

        return build_result

    def _prepare_build(self):
        """Prepare the build environment."""
        from spectrochempy.utils.file import (
            download_testdata,  # Import here to avoid unnecessary delay
        )

        environ["DOC_BUILDING"] = "yes"

        self._version, self._doc_version = self._determine_version()
        download_testdata()
        self._make_dirs()

        # APIGEN?
        if not self.settings["noapi"]:
            self._apigen()

        if not self.settings["nosync"]:
            self._sync_notebooks()

        # Get previous versions and pass them to the template
        previous_versions = self._get_previous_versions()
        environ["PREVIOUS_VERSIONS"] = ",".join(previous_versions)

        # Set Sphinx configuration options based on settings
        environ["SPHINX_NOEXEC"] = "1" if self.settings["noexec"] else "0"

    def _run_sphinx_build(self, builder):
        """Run the Sphinx build process."""
        version = self._version
        doc_version = self._doc_version
        BUILDDIR = DOCREPO / builder

        print(
            f"\n{'-' * 80}\n"
            f"Building {builder.upper()} documentation ({doc_version.capitalize()} "
            f"version : {version})"
            f"\n in {BUILDDIR}"
            f"\n{'-' * 80}"
        )
        srcdir = confdir = DOCS
        outdir = f"{BUILDDIR}/latest"
        doctreesdir = f"{DOCTREES}/latest"

        sp = Sphinx(
            str(srcdir),
            str(confdir),
            str(outdir),
            str(doctreesdir),
            builder,
            warningiserror=self.settings["warningiserror"],
            parallel=self.settings["jobs"],
            verbosity=self.settings["verbosity"],
        )
        return sp.build()

    def _post_build(self, builder):
        """Post-build actions."""
        doc_version = self._doc_version
        BUILDDIR = DOCREPO / builder
        sh(f"cp -r {BUILDDIR}/latest/ {BUILDDIR}")

        if doc_version == "stable":
            # Move stable version to "stable/" subdirectory and populate the root directory
            stable_dir = BUILDDIR / "stable"
            sh(f"rm -rf {stable_dir}")
            sh(f"cp -r {BUILDDIR}/latest/ {stable_dir}")

        print(
            f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
            f"are in {BUILDDIR}/{doc_version} and copied in {BUILDDIR}."
        )

        del environ["DOC_BUILDING"]

    def clean(self):
        """Clean/remove the built documentation."""
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
        """Create a zip file of tutorials."""
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
            """Zip the directory."""
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

    def sync_nb(self):
        """Perform only the pairing of .py and notebooks."""
        if self.settings["delnb"]:
            self._delnb()  # erase nb before starting
        self._sync_notebooks()


# This two classes can be used from spectrochempy.utils.system when all spectrochempy will not be imported in this case
class sh:
    """Utility to run subprocess run command as if they were functions."""

    def __getattr__(self, command):
        return _ExecCommand(command)

    def __call__(self, script, silent=False):
        """Run a shell script."""
        # Ensure that the script input is validated or sanitized before this point
        safe_script = shlex.split(script)
        proc = run(  # noqa: S603
            safe_script, text=True, stdout=PIPE, stderr=STDOUT, check=False
        )

        if not silent:
            print(proc.stdout)  # noqa: T201
        return proc.stdout


class _ExecCommand:
    """Class to execute shell commands."""

    def __init__(self, command):
        self.commands = [command]

    def __call__(self, *args, **kwargs):
        """Execute the command with arguments."""
        args = list(args)
        args[-1] = str(args[-1])  # convert Path to str
        self.commands.extend(args)

        silent = kwargs.pop("silent", False)
        # Ensure that the command input is validated or sanitized before this point
        safe_command = shlex.split(" ".join(self.commands))
        proc = run(  # noqa: S603
            safe_command,
            text=True,
            stdout=PIPE,
            stderr=STDOUT,
            check=False,
        )  # capture_output=True)

        # TODO: handle error codes
        if not silent and proc.stdout:
            print(proc.stdout)  # noqa: T201
        return proc.stdout


sh = sh()


# ======================================================================================
def main():
    """Build documentation for SpectroChemPy."""
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
        "--delnb", "-D", help="delete all ipynb files", action="store_true"
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
        "--no-sync",
        "-Y",
        help="do not sync py and ipynb files",
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
        "--jobs", "-j", default=1, help="number of jobs used by sphinx-build"
    )

    parser.add_argument(
        "--sync-nb", "-S", help="sync .py and notebooks", action="store_true"
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
        delnb=args.delnb,
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


# ======================================================================================
if __name__ == "__main__":
    sys.exit(main())
