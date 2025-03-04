# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201,S603, S605
"""
Documentation Builder for SpectroChemPy.

This script manages the building process of SpectroChemPy's documentation.
It handles:
- Building HTML documentation for current and older versions
- API documentation generation
- Notebook synchronization with their Python counterparts
- Tutorial packaging and distribution
- Multiple version support with Git tags
- Directory structure management for documentation
- Test data downloading and setup

Some code is copied from the Pandas project's documentation builder.

Commands
--------
html :
    Build HTML documentation
clean :
    Remove built documentation and clean notebooks
sync-nb :
    Synchronize notebooks with Python files
tutorials :
    Create zip archive of tutorials

Options
-------
--del-nb, -D : Delete all ipynb files
--no-api, -A : Skip API regeneration
--no-exec, -E : Do not execute notebooks
--no-sync, -Y : Skip py/ipynb synchronization
--jobs, -j : Number of parallel jobs
--tag-name, -T : Build docs for specific version
--clear, -C : Clear html directory
--warning-is-error, -W : Fail if warnings are raised
--single-doc : Build a single document
--directory, -d : Build a specific directory
--whatsnew : Build only the whatsnew document

Examples
--------
Build HTML docs using all CPU cores:
    python make.py html -j auto

Clean build artifacts:
    python make.py clean

Build docs for specific version:
    python make.py html -T 0.6.10
"""

import argparse
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from contextlib import suppress
from os import environ
from pathlib import Path

from tools.helpers import sh

# Suppress other specific warnings
warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)

warnings.filterwarnings(action="ignore", module="debugpy")
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    action="ignore",
    message="Gallery in version 0.18.0 is not supported by Jupytext",
    category=UserWarning,
)

# Constants for paths and configuration
# PROJECTNAME = "spectrochempy"
# REPO_URI = f"spectrochempy/src/{PROJECTNAME}"
# API_GITHUB_URL = "https://api.github.com"
# URL_SCPY = "www.spectrochempy.fr"

# Base directory structure
DOCS = Path(__file__).parent
TEMPLATES = DOCS / "_templates"
STATIC = DOCS / "_static"
PROJECT = DOCS.parent
BUILDDIR = PROJECT / "build"
DOCTREES = BUILDDIR / "~doctrees"
HTML = BUILDDIR / "html"
DOWNLOADS = HTML / "downloads"
TEMPDIRS = PROJECT.parent / "tempdirs"

ON_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true"


# ======================================================================================
# Class BuildOldTagDocs
# ======================================================================================
class BuildOldTagDocs:
    """
    Manages building documentation for older versions of SpectroChemPy.

    This class handles working with old versions of the documentation by managing
    git worktrees, virtual environments, and package installation.

    Attributes
    ----------
    tagname : str
        The git tag name to build documentation for
    verbose : bool
        Whether to print verbose output
    workingdir : Path
        Path to temporary working directory
    """

    def __init__(self, **kwargs):
        self.tagname = kwargs.get("tagname")
        if not self.tagname:
            raise ValueError("Please provide a tag name.")

        self.verbose = kwargs.get("verbose")

        # Create a temporary working directory
        self.workingdir = self._create_temp_directory()

    def _create_temp_directory(self):
        """
        Create temporary working directory in current location.

        Returns
        -------
        Path
            Path to the created temporary directory
        """
        TEMPDIRS.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="scp_", dir=TEMPDIRS))

    def import_scp_version(self):
        """
        Import a specific version of spectrochempy from a git tag.

        This method:
        1. Creates a git worktree for the specified tag
        2. Sets up a virtual environment
        3. Installs the package and dependencies
        4. Imports and returns the module

        Returns
        -------
        module or None
            The imported spectrochempy module if successful, None otherwise

        Raises
        ------
        RuntimeError
            If git repository or tag is not found
        FileNotFoundError
            If setup.py is missing
        """
        workingdir = self.workingdir
        tagname = self.tagname

        try:
            if self.verbose:
                # Debug prints
                print(f"Project directory: {PROJECT}")
                print(f"Temporary directory: {workingdir}")
                print(f"Working with tag: {tagname}")
            self.uv = uv = shutil.which("uv")
            print("UV : ", uv)
            # Ensure we're in a git repository and tag exists
            current_dir = os.getcwd()
            try:
                os.chdir(PROJECT)  # Change to project directory for git commands
                if not Path(".git").exists():
                    raise RuntimeError(f"{PROJECT} is not a git repository")

                # Check if tag exists locally
                tag_exists = sh("git tag -l " + tagname, silent=True).strip()
                if not tag_exists:
                    print("Tag not found locally, trying to fetch...")
                    sh("git fetch origin --tags", silent=False)
                    tag_exists = sh("git tag -l " + tagname, silent=True).strip()
                    if not tag_exists:
                        raise RuntimeError(f"Tag {tagname} not found in repository")

                print(f"Found tag: {tag_exists}")

                # Create a worktree from the local repository at the specific tag
                sh(
                    f"git worktree add --detach {str(workingdir)} {tagname}",
                    silent=False,
                )

            finally:
                os.chdir(current_dir)  # Restore original directory

            # Verify directory contents
            print(f"Contents of {workingdir}:")
            sh(f"ls -la {workingdir}", silent=False)

            # Check for setup.py in different locations
            install_dir = workingdir
            if not (install_dir / "setup.py").exists():
                install_dir = install_dir / "scp"
                print(f"Checking alternate location: {install_dir}")
                if not install_dir.exists():
                    print(f"scp directory not found at {install_dir}")
                if not (install_dir / "setup.py").exists():
                    raise FileNotFoundError(
                        f"No setup.py found in version {tagname} (checked {workingdir} and {install_dir})"
                    )

            print(f"Installing from: {install_dir}")

            # We will use uv for the installation of the requirements
            # as it is faster than pip
            # Install it if not already done
            try:
                import uv
            except ImportError:
                os.system(f"{sys.executable} install uv")

            # get the full pâth of the installed uv command
            self.uv = uv = shutil.which("uv")
            print("UV : ", uv)
            # Install requirements and package in development mode
            print("Installing requirements...")
            req_file = DOCS / "tools" / "req_old_scpy.txt"
            if not req_file.exists():
                raise FileNotFoundError(f"Requirements file not found at {req_file}")
            os.system(f"{uv} pip install -r {req_file} --force-reinstall")
            print("Installing package in development mode...")
            os.system(f"{uv} pip install -e {str(install_dir)} --no-deps")

            # Add package directory to Python path
            package_dir = workingdir / "spectrochempy"
            if not package_dir.exists():
                package_dir = install_dir / "spectrochempy"
            if package_dir.exists() and str(package_dir.parent) not in sys.path:
                sys.path.insert(0, str(package_dir.parent))

            # Force reload of any existing spectrochempy module
            if "spectrochempy" in sys.modules:
                del sys.modules["spectrochempy"]

            # Import with error details
            try:
                import spectrochempy as scp

                print(f"Successfully imported spectrochempy version {scp.version}")
                return scp
            except ImportError as e:
                print(f"Import error: {e}")
                print(f"Python path: {sys.path}")
                raise

        except Exception as e:
            print(f"Failed to setup version {tagname}")
            print(f"Error details: {str(e)}")
            if hasattr(e, "stderr"):
                print(f"Error output: {e.stderr}")
            return None

    def cleanup(self):
        """Cleanup the worktree and temporary directory."""
        if hasattr(self, "workingdir") and self.workingdir.exists():
            try:
                # First change to project directory
                current_dir = os.getcwd()
                os.chdir(PROJECT)

                try:
                    # Remove the worktree
                    sh(f"git worktree remove --force {self.workingdir}", silent=True)
                finally:
                    # Always restore original directory
                    os.chdir(current_dir)

                # Remove directory if it still exists
                if self.workingdir.exists():
                    import shutil

                    shutil.rmtree(self.workingdir)
            except Exception as e:
                print(f"Cleanup failed: {e}")

    def __del__(self):
        """Attempt cleanup on object destruction."""
        self.cleanup()

    def restore_original_version(self):
        """Restore the original version of spectrochempy."""
        # if we are building in githu, we do not need to restore the original version
        if ON_GITHUB:
            return

        try:
            # Install the original and latest version from the base directory
            sh(
                f'{self.uv} -m pip install -e ".[dev, docs]" --force-reinstall',
                silent=False,
            )

            # Force reload
            if "spectrochempy" in sys.modules:
                del sys.modules["spectrochempy"]

        except Exception as e:
            print(f"Failed to restore original version {self.original_version}: {e}")


# ======================================================================================
# Class BuildDocumentation
# ======================================================================================
class BuildDocumentation:
    """
    Main documentation builder for SpectroChemPy.

    Handles all aspects of documentation building including configuration,
    notebook synchronization, and HTML generation.
    """

    def __init__(self, **kwargs):
        # Initialize the BuildDocumentation class with settings.
        self.settings = settings = self._init_settings(kwargs)

        # DOCUMENTATION SRC PATH
        # They depends if we are building the documentation
        #  for the latest version or for an older one
        self.tagname = settings["tagname"]
        if self.tagname:
            self.SRC = SRC = Path(kwargs.get("workingdir")) / "docs"
            self.PROJECT_SOURCES = Path(kwargs.get("workingdir")) / "spectrochempy"
        else:
            self.SRC = SRC = DOCS / "sources"
            self.PROJECT_SOURCES = PROJECT / "src" / "spectrochempy"
        self.GETTINGSTARTED = SRC / "gettingstarted"
        self.DEVGUIDE = SRC / "devguide"
        self.REFERENCE = SRC / "reference"

        # Generated by sphinx
        self.API = self.REFERENCE / "generated"
        self.DEV_API = self.DEVGUIDE / "generated"
        self.GALLERY = self.GETTINGSTARTED / "examples" / "gallery"

        # Set environmetnt variables for sphinx
        environ["SPHINX_NOEXEC"] = "1" if settings["noexec"] else "0"
        self.singledoc = settings["singledoc"]
        self.directory = settings["directory"]

        if self.directory:
            self.directory = self._validate_directory(self.directory)
            os.environ["SPHINX_PATTERN"] = f"dir:{self.directory}"
        elif self.singledoc:
            self.singledoc = self._single_doc(self.singledoc)
            os.environ["SPHINX_PATTERN"] = self.singledoc
        elif settings["noapi"]:
            os.environ["SPHINX_PATTERN"] = "noapi"
        elif settings["whatsnew"]:
            os.environ["SPHINX_PATTERN"] = "whatsnew"

    def _validate_directory(self, directory):
        """Validate and return normalized directory path."""
        dir_path = self.SRC / directory
        if not dir_path.is_dir():
            raise ValueError(f"Directory {directory} not found in sources")
        return directory

    def _init_settings(self, kwargs):
        # Initialize settings from keyword arguments.
        # Parameters:
        # kwargs : dict - Keyword arguments to initialize settings from
        # Returns: dict - Dictionary of initialized settings

        return {
            "delnb": kwargs.get("delnb", False),
            "noapi": kwargs.get("noapi", False),
            "noexec": kwargs.get("noexec", False),
            "nosync": kwargs.get("nosync", False),
            "clear": kwargs.get("clear", False),
            "tutorials": kwargs.get("tutorials", False),
            "verbosity": kwargs.get("verbosity", 0),
            "jobs": self._get_jobs(kwargs.get("jobs", "auto")),
            "warningiserror": kwargs.get("warning_is_error", False),
            "whatsnew": kwargs.get("whatsnew", False),
            "tagname": kwargs.get("tagname", None),
            "singledoc": kwargs.get("singledoc", None),
            "directory": kwargs.get("directory", None),
        }

    def _single_doc(self, singledoc):
        # Make sure the provided value for --single is a path to an existing
        # .rst/.ipynb file, or a spectrochempy object that can be imported.
        # For example, citing.rst or spectrochempy.IRIS. For the latter,
        # return the corresponding file path
        # (e.g. reference/generated/spectrochempy.IRIS.rst).
        # adapted from pandas

        extension = os.path.splitext(singledoc)[-1]
        if extension in (".rst", ".ipynb"):
            if (self.SRC / singledoc).exists():
                return singledoc
            raise FileNotFoundError(f"File {str(self.SRC / singledoc)} not found")

        if singledoc.startswith("spectrochempy."):
            try:
                import spectrochempy

                obj = spectrochempy
                for name in singledoc.split(".")[1:]:
                    obj = getattr(obj, name)
            except AttributeError as err:
                raise ImportError(f"Could not import {singledoc}") from err
            else:
                # delete the eventual already generated entry
                if Path(self.API / f"{singledoc}.rst").exists():
                    Path(self.API / f"{singledoc}.rst").unlink()
                return singledoc[len("spectrochempy.") :]
        else:
            raise ValueError(
                f"--single={singledoc} not understood. "
                "Value should be a valid path to a .rst or .ipynb file, "
                "or a valid spectrochempy object "
                "(e.g. citing.rst or spectrochempy.IRIS)"
            )

    @staticmethod
    def _get_jobs(jobs):
        # Get the number of jobs to use for building the documentation.
        # Parameters:
        # jobs : str - Number of jobs to use, or 'auto' to use all CPU cores
        # Returns: int - Number of jobs to use

        if jobs == "auto":
            return mp.cpu_count()
        try:
            return int(jobs)
        except ValueError:
            print("Error: --jobs argument must be an integer or 'auto'")
            return 1

    def _delnb(self):
        # Remove all Jupyter notebook files.

        for nb in self.SRC.rglob("**/*.ipynb"):
            sh.rm(nb)
        for nbch in self.SRC.glob("**/.ipynb_checkpoints"):
            sh(f"rm -r {nbch}")
        print(f"Removed all ipynb files in {self.SRC}")

    def _sync_notebooks(self):
        # Synchronize notebook and Python script pairs.
        # Finds and synchronizes all notebook/script pairs using jupytext,
        # with error handling for individual files.

        print(f"\n{'-' * 80}\nSync *.py and *.ipynb using jupytext\n{'-' * 80}")
        if self.settings["nosync"] or self.settings["whatsnew"]:
            print("Skipping notebook synchronization as option --no-sync is set")
            return

        if self.singledoc:
            if not self.singledoc.endswith(".ipynb"):
                print("Nothing to sync")
                return
            print(f"Syncing only {self.singledoc}")
            self._sync_notebook_pair(self.SRC / self.singledoc)
            return

        for item in self._get_notebook_files():
            if self.directory and str(self.directory) not in str(item):
                continue
            try:
                if self.settings["delnb"]:
                    item.with_suffix(".ipynb").unlink()
                self._sync_notebook_pair(item)
            except Exception as e:
                print(f"Failed to sync {item}: {e}")

    def _get_notebook_files(self):
        # Get a set of notebook files to be synchronized.
        # Returns: set - Set of notebook files to be synchronized

        pyfiles = set()
        py = list(self.SRC.glob("**/*.py"))
        py.extend(list(self.SRC.glob("**/*.ipynb")))

        for f in py[:]:
            # Do not consider some files
            if (
                "generated" in f.parts
                or ".ipynb_checkpoints" in f.parts
                or "gallery" in f.parts
                or "examples" in f.parts
                or "sphinxext" in f.parts
            ) or f.name in ["conf.py", "make.py", "apigen.py"]:
                continue
            # Add only the full path without suffix
            pyfiles.add(f.with_suffix(""))

        return pyfiles

    @staticmethod
    def _sync_notebook_pair(item):
        # Synchronize a pair of notebook and script files.
        # Parameters:
        # item : Path - Path to the notebook or script file to synchronize

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

    def _determine_version(self):
        # Determine the version of the documentation to build.
        # Returns: tuple - Contains (version, last_tag, version_type)
        # where:
        # - version : str - The version number
        # - last_tag : str - The previous release tag
        # - version_type : str - One of 'latest' or <tag>

        from spectrochempy.api import version

        last_tag = self._get_previous_tag() if not self.tagname else None
        if self.tagname is not None:
            return self.tagname, last_tag, self.tagname
        return version, last_tag, "latest" if "dev" in version else last_tag

    @staticmethod
    def _get_previous_tag():
        # Get the previous tag from the git repository.
        # Returns: str - The previous release tag

        sh("git fetch --tags", silent=True)
        rev = sh("git rev-list --tags --max-count=1", silent=True)
        result = sh(f"git describe --tags {rev}", silent=True)
        return result.strip()

    def _make_dirs(self):
        # Create the directories required to build the documentation.

        doc_version = self._doc_version

        # Create regular directories if they do not exist already.
        build_dirs = [
            BUILDDIR,
            DOCTREES,
            HTML,
            DOCTREES / doc_version,
            HTML / doc_version,
            # DOWNLOADS,
        ]
        for d in build_dirs:
            Path.mkdir(d, parents=True, exist_ok=True)

    def _get_previous_versions(self):
        # Get a list of previous versions from the HTML directory.
        # Returns: list - List of previous versions

        versions = []
        version_pattern = re.compile(r"^(\d+\.\d+\.\d+)$")
        for item in HTML.iterdir():
            if item.is_dir() and version_pattern.match(item.name):
                versions.append(item.name)
        return versions

    def _make_docs(self):
        # Simplified documentation building process.
        # Returns: int - Sphinx build result

        self._prepare_build()
        build_result = self._run_sphinx_build()
        self._post_build()
        return build_result

    def _prepare_build(self):
        # Prepare the build environment.

        if self.tagname:
            print(
                f"\n{'-' * 80}\n"
                f"Preparing to build the older documentation for tag {self.tagname}"
                f"\n{'-' * 80}"
            )
        else:
            print(
                f"\n{'-' * 80}\n"
                "Preparing to build the documentation - load spectrochempy and testdata"
                f"\n{'-' * 80}"
            )

        if not self.settings["whatsnew"]:
            print("Loading spectrochempy and downloading test data...")

            from spectrochempy.utils.file import download_testdata

            # Download the test data
            download_testdata()

        # Determine the version of the documentation to build
        self._version, self._last_release, self._doc_version = self._determine_version()

        # Create the directories required for building the documentation
        self._make_dirs()

        # Get previous versions and save them in a json file to b use by the versions.js scipt
        previous_versions = self._get_previous_versions()

        # Eventually clean the build target directory
        # Clean the files and directories that are not version folders or latest
        if self.settings["clear"]:
            for item in HTML.iterdir():
                if item.is_dir() and item.name not in previous_versions:
                    shutil.rmtree(item, ignore_errors=True)
                    print(f"Removed directory: {item}")
                elif item.is_file() and item.name not in previous_versions:
                    item.unlink()
                    print(f"Removed file: {item}")

        # Sync the notebooks with the Python scripts
        if not self.settings["nosync"]:
            self._sync_notebooks()

        # Set the environment variables for the Sphinx build
        environ["DOC_BUILDING"] = "yes"
        environ["PREVIOUS_VERSIONS"] = ",".join(previous_versions)
        environ["LAST_RELEASE"] = self._last_release if self._last_release else ""

        return

    def _run_sphinx_build(self):
        # Run the Sphinx build process.
        # Returns: int - Sphinx build result

        from sphinx.application import Sphinx
        from sphinx.errors import ExtensionError

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

        version = self._version
        doc_version = self._doc_version

        # Copy custom CSS file to the _static/css directory
        custom_css_src = (
            PROJECT / "src" / "spectrochempy" / "data" / "css" / "custom.css"
        )
        custom_css_dest = STATIC / "css" / "custom.css"
        custom_css_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(custom_css_src, custom_css_dest)
        print(f"Copied custom CSS from {custom_css_src} to {custom_css_dest}")

        print(
            f"\n{'-' * 80}\n"
            f"Building HTML documentation ({doc_version.capitalize()} "
            f"version : {version})"
            f"\n in {HTML}"
            f"\n{'-' * 80}"
        )

        environ["SPHINX_SRCDIR"] = srcdir = str(self.SRC)
        environ["SPHINX_CONFDIR"] = confdir = str(DOCS)
        environ["SOURCES"] = str(self.PROJECT_SOURCES)

        outdir = f"{HTML}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"

        sp = Sphinx(
            srcdir,
            confdir,
            outdir,
            doctreesdir,
            "html",
            warningiserror=self.settings["warningiserror"],
            parallel=self.settings["jobs"],
            verbosity=self.settings["verbosity"],
        )

        class SafeEventEmitter:
            def __init__(self, original_emit):
                self.original_emit = original_emit

            def __call__(self, *args, **kwargs):
                try:
                    return self.original_emit(*args, **kwargs)
                except ExtensionError as e:
                    if "embed_code_links" in str(e):
                        print("Note: Ignoring embed_code_links error to continue build")
                        return None
                    raise

        # Replace the event emitter with our safe version
        sp.events.emit = SafeEventEmitter(sp.events.emit)

        try:
            sp.build()
            return 0
        except Exception as e:
            print(f"Warning: Build encountered an error: {e}")
            if "build-finished" in str(e):
                print("Build completed despite embed_code_links error")
                return 0
            raise

    def _post_build(self):
        # Post-build actions.

        doc_version = self._doc_version

        # Check if the source directory exists and is not empty
        source_dir = HTML / doc_version
        if source_dir.exists() and any(source_dir.iterdir()):
            # Copy all files, including hidden ones, from source_dir to HTML
            # except if we are building an oldest version or version is dirty
            if not self.tagname:
                for item in source_dir.iterdir():
                    dest = HTML / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                print(f"Copied contents of {source_dir} to {HTML}/")
        else:
            print(f"Warning: Source directory {source_dir} does not exist or is empty")

        # Remove it if doc_version is 'latest' as all content is in the parent directory
        if doc_version == "latest" and source_dir.exists():
            shutil.rmtree(source_dir)
            print(f"Removed directory {source_dir}")

        # Get all version directories
        versions = []
        version_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        for item in HTML.iterdir():
            if item.is_dir() and version_pattern.match(item.name):
                versions.append(item.name)

        versions.sort(reverse=True)  # Sort in descending order
        versions_str = ",".join(versions)

        # Update layout.html to include latest versions list
        layout_template = TEMPLATES / "layout.html"
        with open(layout_template) as f:
            content = f.read()
            content = content.replace(
                "data-versions=\"{{ os.environ.get('PREVIOUS_VERSIONS', '') }}\"",
                f'data-versions="{versions_str}"',
            )

        # Update also in each version directory
        for version_dir in HTML.glob("[0-9]*.[0-9]*.[0-9]*"):
            target_dir = version_dir / "_templates"
            target_dir.mkdir(exist_ok=True)
            target_file = target_dir / "layout.html"

            with open(target_file, "w") as f:
                f.write(content)

        # Remove the environment variables
        del environ["DOC_BUILDING"]
        del environ["PREVIOUS_VERSIONS"]
        del environ["SPHINX_NOEXEC"]
        if "SPHINX_PATTERN" in environ:
            del environ["SPHINX_PATTERN"]

    # COMMANDS
    # ----------------------------------------------------------------------------------

    def clean(self):
        """Clean/remove the built documentation."""
        print(f"\n{'-' * 80}\nCleaning\n{'-' * 80}")

        for doc_version in ["latest", "tagged", "dirty"]:
            shutil.rmtree(HTML / doc_version, ignore_errors=True)
            print(f"removed {HTML / doc_version}")
            shutil.rmtree(DOCTREES / doc_version, ignore_errors=True)
            print(f"removed {DOCTREES / doc_version}")

        shutil.rmtree(self.API, ignore_errors=True)
        print(f"removed {self.API}")
        shutil.rmtree(self.DEV_API, ignore_errors=True)
        print(f"removed {self.DEV_API}")
        shutil.rmtree(self.GALLERY, ignore_errors=True)
        print(f"removed {self.GALLERY}")

        self._delnb()

    def html(self):
        """Generate HTML documentation."""
        return self._make_docs()

    def tutorials(self):
        """Create a zip file of tutorials."""
        # Make tutorials.zip
        print(f"\n{'-' * 80}\nMake tutorials.zip\n{'-' * 80}")

        # Clean notebooks output
        for nb in DOCS.rglob("**/*.ipynb"):
            # This will erase all notebook output
            sh(
                f"jupyter nbconvert "
                f"--ClearOutputPreprocessor.enabled=True --inplace {nb}",
                silent=True,
            )

        # Make zip of all ipynb
        def _zipdir(path, dest, ziph):
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
        _zipdir(self.SRC, "notebooks", zipf)
        _zipdir(self.GALLERY / "auto_examples", Path("notebooks") / "examples", zipf)
        zipf.close()

        sh(
            f"mv ~notebooks.zip "
            f"{DOWNLOADS}/{self.doc_version}-spectrochempy-notebooks.zip"
        )

    def sync_nb(self):
        """Perform only the pairing of .py and notebooks."""
        if self.settings["delnb"]:
            self._delnb()  # Erase nb before starting
        self._sync_notebooks()

    def linkcheck(self):
        """Check the links in the documentation."""
        from sphinx.cmd.build import build_main

        srcdir = str(DOCS)
        args = [
            "-b",
            "linkcheck",  # Use linkcheck builder
            "-W",  # Treat warnings as errors
            "-n",
            "-q",  # Run in nit-picky mode, quietly
            str(srcdir),  # Source directory
            str(BUILDDIR / "linkcheck"),  # Output directory
        ]
        return build_main(args)


# ======================================================================================
def main():
    """
    Command-line interface for the documentation builder.

    Parses command line arguments and orchestrates the documentation
    build process based on provided options.

    Returns
    -------
    int
        0 for success, 1 for failure

    Notes
    -----
    Supported commands include:
    - html : Build HTML documentation
    - clean : Remove built documentation
    - sync-nb : Synchronize notebooks
    - linkcheck : Check the links in the documentation

    Use -h/--help to see all available options.
    """
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
        "--no-sync", "-Y", help="do not sync py and ipynb files", action="store_true"
    )

    parser.add_argument(
        "--upload-tutorials", "-Z", help="zip and upload tutorials", action="store_true"
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
        "--single-doc",
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

    parser.add_argument(
        "--clear", "-C", help="clear the html directory", action="store_true"
    )

    parser.add_argument(
        "--tag-name", "-T", type=str, help="Git tag to read from to regenerate old docs"
    )

    parser.add_argument(
        "--directory",
        "-d",
        metavar="DIR",
        type=str,
        default=None,
        help="directory to build (e.g., gettingstarted/install)",
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

    if args.tag_name and args.command != "html":
        print("The --tagname option is only valid with the html command.")
        return 1

    if args.directory and not args.directory.endswith("/"):
        args.directory += "/"

    if not args.tag_name:
        # build the documentation for the latest version
        build = BuildDocumentation(
            delnb=args.del_nb,
            noapi=args.no_api,
            noexec=args.no_exec,
            nosync=args.no_sync,
            clear=args.clear,
            tutorials=args.upload_tutorials,
            verbosity=args.verbosity,
            jobs=args.jobs,
            warningiserror=args.warning_is_error,
            whatsnew=args.whatsnew,
            singledoc=args.single_doc,
            directory=args.directory,
        )

        buildcommand = getattr(build, args.command)
        res = buildcommand()

    else:
        # build the documentation for an older version
        build_old = None
        try:
            build_old = BuildOldTagDocs(tagname=args.tag_name, verbose=True)
            scp = build_old.import_scp_version()
            if scp is None:
                return 1

            # Create BuildDocumentation instance directly (don't import it)
            build = BuildDocumentation(
                jobs=args.jobs,
                tagname=build_old.tagname,
                workingdir=build_old.workingdir,
            )

            buildcommand = getattr(build, args.command)
            res = buildcommand()

        finally:
            # Always attempt cleanup and restore original version
            if build_old is not None:
                build_old.restore_original_version()
                build_old.cleanup()

    # return 0 if successful, 1 if failed
    if res is None:
        res = 0
    return res


# ======================================================================================
if __name__ == "__main__":
    sys.exit(main())
