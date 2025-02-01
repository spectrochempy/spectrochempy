# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: T201,S603
"""SpectroChemPy documentation build configuration file."""

import inspect
import os
import pathlib
import sys
import warnings
from datetime import datetime

import spectrochempy  # isort:skip

# set a filename and default folder by default for notebook which have file dialogs
os.environ["TUTORIAL_FILENAME"] = "wodger.spg"
os.environ["TUTORIAL_FOLDER"] = "irdata/subdir"


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation general, use os.path.abspath to make it absolute, like shown
# here: sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your
# custom ones.

# hack to make import
sys._called_from_sphinx = True

# Sphinx Extensions
source = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(source, "docs", "sphinxext"))

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# General information about the project.

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = spectrochempy.application.version  # .split('+')[0]
release = version.split("+")[0]
project = f"SpectroChemPy v{version}"
copyright = spectrochempy.application.copyright

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d, %Y"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_templates",
    "_static",
    "**.ipynb_checkpoints",
    "~temp",
    "generated",
    "gettingstarted/examples/gallery/**/*.py",
    "gettingstarted/examples/gallery/**/*.ipynb",
    "**.md5",
]

# The reST default role (used for this markup: `text` ) to use for all
# documents.
default_role = "py:obj"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# Show Todo box
todo_include_todos = True

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "navigation_depth": 5,
    "sticky_navigation": True,
    "prev_next_buttons_location": "both",
}

# Add any paths that contain custom themes here, relative to this directory.

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None
# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None
# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

html_logo = "_static/scpy.png"
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.

html_favicon = "_static/scpy.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/spectrochempy.css",
]
# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""
# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = "spectrochempydoc"

trim_doctests_flags = True

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

html_context = {
    "current_version": "latest" if ("dev" in version) else "stable",
    "release": release,
    "base_url": "..",
    "versions": (
        ("latest", '/latest/index.html"'),
        ("stable", "/stable/index.html"),
    ),
    # This is for the citing page
    "version": release,
    "bibversion": "{" + release + "}",
    "bibmonth": "{" + f"{datetime.today().month}" + "}",
    "year": f"{datetime.today().year}",
    "bibyear": "{" + f"{datetime.today().year}" + "}",
}

# Sphinx-gallery ---------------------------------------------------------------
# Generate the plots for the gallery
from sphinx_gallery.sorting import FileNameSortKey

example_source_dir = "../src/spectrochempy/examples"
example_generated_dir = "gettingstarted/examples/gallery"

sphinx_gallery_conf = {
    "plot_gallery": "True",
    "doc_module": "spectrochempy",
    # Source example files in separate directory
    "examples_dirs": [
        f"{example_source_dir}/core",
        f"{example_source_dir}/processing",
        f"{example_source_dir}/analysis",
    ],
    # Generated RST files in generated directory
    "gallery_dirs": [
        f"{example_generated_dir}/auto_examples_core",
        f"{example_generated_dir}/auto_examples_processing",
        f"{example_generated_dir}/auto_examples_analysis",
    ],
    "backreferences_dir": f"{example_generated_dir}/backreferences",
    "reference_url": {
        "spectrochempy": None,
    },
    "show_memory": False,
    "thumbnail_size": (400, 400),
    "abort_on_example_error": False,
    "only_warn_on_example_error": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "expected_failing_examples": [],
    "download_all_examples": False,
    "pypandoc": True,
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
    "image_scrapers": ("matplotlib",),
    "filename_pattern": "/plot",
    "ignore_pattern": "__init__\\.py",
    "min_reported_time": 0,
    "show_signature": False,  # Disable the signature if it's causing issues
}

suppress_warnings = [
    "sphinx_gallery",
]

# nbsphinx ---------------------------------------------------------------------

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'jpg', 'png'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_exclude_patterns = [
    f"{example_generated_dir}/*",
]

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = "always"
nbsphinx_allow_errors = True
nbsphinx_timeout = 90
nbsphinx_prolog = """
"""
nbsphinx_epilog = """
"""

# Use this kernel instead of the one stored in the notebook metadata:
nbsphinx_kernel_name = "python3"

# Support for notebook formats other than .ipynb
nbsphinx_custom_formats = {
    # ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
    # ".md": ["jupytext.reads", {"fmt": "Rmd"}],
}

# Configure sphinxcontrib-bibtex

bibtex_bibfiles = ["reference/bibliography.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
bibtex_cite_id = "{key}"

# configuration for intersphinx --------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "SciPy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "traitlets": ("https://traitlets.readthedocs.io/en/stable/", None),
}

# linkcode -----------------------------------------------------------------------------


def linkcode_resolve(domain, info):
    # Resolve function for the linkcode extension.
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(spectrochempy.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        fs = find_source()
        filename = "spectrochempy/{}#L{}-L{}".format(*fs)
    except TypeError:
        return None
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "master"
    return f"https://github.com/spectrochempy/spectrochempy/blob/{tag}/{filename}"


# Autosummary --------------------------------------------------------------------------

pattern = os.environ.get("SPHINX_NOAPI")
include_api = pattern is None
autosummary_generate = True if include_api else ["index"]

autodoc_typehints = "none"
napoleon_use_param = False
napoleon_use_rtype = False

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False
numpydoc_use_plots = True

autoclass_content = "both"
# Both the class’ and the __init__ method’s docstring are concatenated and inserted.

autodoc_default_options = {"autosummary": include_api}
autodoc_class_signature = "mixed"
exclusions = (
    "_*",
    "add_traits",
    "class_config_rst_doc",
    "class_config_section",
    "class_get_help",
    "class_own_trait_events",
    "class_own_traits",
    "class_print_help",
    "class_trait_names",
    "class_traits",
    "clear_instance",
    "cross_validation_lock",
    "document_config_options",
    "flatten_flags",
    "generate_config_file",
    "has_trait",
    "hold_trait_notifications",
    "initialize_subcommand",
    "initialized",
    "instance",
    "json_config_loader_class",
    "launch_instance",
    "load_config_file",
    "notify_change",
    "observe",
    "on_trait_change",
    "parse_command_line",
    "print_alias_help",
    "print_description",
    "print_examples",
    "print_flag_help",
    "print_help",
    "print_subcommands",
    "print_version",
    "python_config_loader_class",
    "section_names",
    "set_trait",
    "setup_instance",
    "trait_events",
    "trait_metadata",
    "trait_names",
    "trait",
    "unobserve_all",
    "unobserve",
    "update_config",
    "with_traceback",
)

# Add newline at end of files
rst_epilog = "\n"


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Determine whether to skip a member during autodoc generation."""
    doc = bool(obj.__doc__ is not None and "#NOT_IN_DOC" not in obj.__doc__)

    exclude = name in exclusions or "trait" in name or name.startswith("_") or not doc
    return skip or exclude


def shorter_signature(app, what, name, obj, options, signature, return_annotation):
    """Prevent displaying self in signature."""
    if what == "data":
        signature = "(dataset)"
        what = "function"

    if what not in ("function", "method", "data") or signature is None:
        return None  # removed"class",

    import re

    new_sig = signature

    if inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj):
        sig_obj = obj if not inspect.isclass(obj) else obj.__init__
        sig_re = r"\((self|cls)?,?\s*(.*?)\)\:"
        try:
            new_sig = " ".join(
                re.search(sig_re, inspect.getsource(sig_obj), re.S)
                .group(2)
                .replace("\n", "")
                .split()
            )
            new_sig = "(" + new_sig + ")"

        except Exception:
            print(sig_obj)  # noqa: T201

    return new_sig, return_annotation


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.

    Copied from site:
    https://ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/
    """
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def setup(app):
    app.connect("source-read", rstjinja)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("autodoc-process-signature", shorter_signature)
    app.add_css_file("css/spectrochempy.css")  # also can be a full URL
    app.connect("builder-inited", on_builder_inited)
    app.connect("build-finished", on_builder_finished)


def on_builder_inited(app):
    """Actions to perform when the builder is initialized."""
    from spectrochempy.utils.file import download_testdata

    # Set environment variable to indicate that we are building the docs
    # (necessary for spectrochempy to bypass dialogs and interactive plots)
    print("Set DOC_BUILDING environment variable")
    os.environ["DOC_BUILDING"] = "yes"

    # Download test data
    print(f"\n{'-' * 80}\nDownload test data\n{'-' * 80}")
    download_testdata()

    # Generate API
    apigen()

    # Sync notebooks
    sync_notebooks()


def on_builder_finished(app, exception):
    """Actions to perform when the builder is finished."""
    # Remove environment variable
    print("Remove DOC_BUILDING environment variable")
    os.environ.pop("DOC_BUILDING", None)


def apigen():
    """Regenerate the reference API list."""
    print(f"\n{'-' * 80}\nRegenerate the reference API list\n{'-' * 80}")
    Apigen()


def sync_notebooks():
    """Synchronize notebooks."""
    import shlex
    from pathlib import Path
    from subprocess import PIPE
    from subprocess import STDOUT
    from subprocess import run

    print(f"\n{'-' * 80}\nSynchronize notebooks\n{'-' * 80}")

    DOCS = Path(__file__).parent
    SRC = DOCS

    pyfiles = set()
    py = list(SRC.glob("**/*.py"))
    py.extend(list(SRC.glob("**/*.ipynb")))

    for f in py[:]:
        if (
            "generated" in f.parts
            or ".ipynb_checkpoints" in f.parts
            or "gallery" in f.parts
            or "examples" in f.parts
            or "sphinxext" in f.parts
        ) or f.name in ["conf.py", "make.py", "apigen.py"]:
            continue
        pyfiles.add(f.with_suffix(""))

    for item in pyfiles:
        py = item.with_suffix(".py")
        ipynb = item.with_suffix(".ipynb")
        file_to_pair = py if py.exists() else ipynb

        try:
            command = f"jupytext --sync {file_to_pair}"
            sanitized_command = shlex.split(command)
            result = run(
                sanitized_command,
                text=True,
                stdout=PIPE,
                stderr=STDOUT,
                check=False,
            ).stdout
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


# ======================================================================================
# Class Apigen
# ======================================================================================
class Apigen:
    """Generate api.rst."""

    header = """
.. Generate API reference pages, but don't display these pages in tables.
.. Do not modify directly because this file is automatically generated
.. when the documentation is built.
.. Only classes and methods appearing in __all__ statements are scanned.

:orphan:

.. currentmodule:: spectrochempy
.. autosummary::
   :toctree: generated/

"""

    def __init__(self):
        entries = self.list_entries()
        self.write_api_rst(entries)

    @staticmethod
    def get_packages():
        from spectrochempy.utils.packages import list_packages

        pkgs = list_packages(spectrochempy)
        for pkg in pkgs[:]:
            if pkg.endswith(".api"):
                pkgs.remove(pkg)
        return pkgs

    def get_members(self, obj, objname, alls=None):
        res = []
        members = inspect.getmembers(obj)
        for member in members:
            _name, _type = member
            if _name == "transform":
                pass
            if (
                (alls is not None and _name not in alls)
                or str(_name).startswith("_")
                or not str(_type).startswith("<")
                or "HasTraits" in str(_type)
                or "cross_validation_lock" in str(_name)
                or not (
                    str(_type).startswith("<class")
                    or str(_type).startswith("<function")
                    or str(_type).startswith("<property")
                )
                or "partial" not in str(_type)
            ):
                continue

            if objname != "spectrochempy" and objname.split(".")[1:][0] in [
                "core",
                "analysis",
                "utils",
                "widgets",
            ]:
                continue

            module = ".".join(objname.split(".")[1:])
            module = module + "." if module else ""
            # print(f"{module}{_name}\t\t{_type}")

            res.append(f"{module}{_name}")

            # if str(_type).startswith("<class"):
            #     # find also members in class
            #     klass = getattr(obj, _name)
            #     subres = self.get_members(klass, objname + "." + _name)
            #     res.extend(subres)

        return res

    def list_entries(self):
        from traitlets import import_item

        pkgs = self.get_packages()

        results = []
        for pkg_name in pkgs:
            if pkg_name.startswith("spectrochempy.examples") or pkg_name.startswith(
                "spectrochempy.extern"
            ):
                continue

            pkg = import_item(pkg_name)
            try:
                alls = pkg.__all__

            except AttributeError:
                # warn("This module has no __all__ attribute")
                continue

            if alls == []:
                continue

            res = self.get_members(pkg, pkg_name, alls)
            results.extend(res)

        return results

    def write_api_rst(self, items):
        REFERENCE = pathlib.Path(__file__).parent / "reference"

        with open(REFERENCE / "api.rst", "w") as f:
            f.write(self.header)
            for item in items:
                f.write(f"    {item}\n")
