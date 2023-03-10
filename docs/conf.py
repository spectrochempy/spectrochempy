# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SpectroChemPy documentation build configuration file
"""

#
import inspect
import os
import sys
import warnings
from datetime import datetime

#
import sphinx_rtd_theme  # Theme for the website

#
import spectrochempy as scp  # isort:skip

# set a filename and default folder by default for notebook which have file dialogs
os.environ["TUTORIAL_FILENAME"] = "wodger.spg"
os.environ["TUTORIAL_FOLDER"] = "irdata/subdir"

# set a flag to deactivate TQDM
os.environ["USE_TQDM"] = "No"

#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation general, use os.path.abspath to make it absolute, like shown
# here: sys.path.insert(0, os.path.abspath('.'))

#
# -- General configuration ---------------------------------------------------

#
# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

#
# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your
# custom ones.

#
# hack to make import
sys._called_from_sphinx = True

#
# Sphinx Extensions
source = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(source, "docs", "sphinxext"))
# print (sys.path)
extensions = [
    "nbsphinx",
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
    "autodoc_traitlets",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
]

#
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

#
# The suffix of source filenames.
source_suffix = ".rst"

#
# The encoding of source files.
source_encoding = "utf-8"

#
# The master toctree document.
master_doc = "index"

#
# General information about the project.

#
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

#
version = scp.application.__version__  # .split('+')[0]
release = version.split("+")[0]
project = f"SpectroChemPy v{version}"
copyright = scp.application.__copyright__

#
# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d, %Y"

#
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_templates", "_static", "**.ipynb_checkpoints", "gallery", "~temp"]

#
# The reST default role (used for this markup: `text` ) to use for all
# documents.
default_role = "obj"

#
# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

#
# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

#
# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

#
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

#
# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


#
# Show Todo box
todo_include_todos = True

#
# This is added to the end of RST files - a good place to put substitutions to
# be used globally.

#
rst_epilog = """

.. |scpy| replace:: **SpectroChemPy**

.. |ndarray| replace:: :class:`~numpy.ndarray`

.. |ma.ndarray| replace:: :class:`~numpy.ma.array`

.. |Project| replace:: :class:`~spectrochempy.core.projects.project.Project`

.. |Script| replace:: :class:`~spectrochempy.core.dataset.scripts.Script`

.. |NDArray| replace:: :class:`~spectrochempy.core.dataset.ndarray.NDArray`

.. |NDDataset| replace:: :class:`~spectrochempy.core.dataset.nddataset.NDDataset`

.. |Coord| replace:: :class:`~spectrochempy.core.dataset.ndcoord.Coord`

.. |CoordSet| replace:: :class:`~spectrochempy.core.dataset.ndcoordset.CoordSet`

.. |NDIO| replace:: :class:`~spectrochempy.core.dataset.ndio.NDIO`

.. |NDMath| replace:: :class:`~spectrochempy.core.dataset.ndmath.NDMath`

.. |Meta| replace:: :class:`~spectrochempy.core.dataset.ndmeta.Meta`

.. |NDPlot| replace:: :class:`~spectrochempy.core.dataset.ndplot.NDPlot`

.. |Unit| replace:: :class:`~spectrochempy.core.units.units.Unit`

.. |Quantity| replace:: :class:`~spectrochempy.core.units.units.Quantity`

.. |Axes| replace:: :class:`~matplotlib.Axes`

"""

#
# -- Options for HTML output ---------------------------------------------------

#
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

#
html_theme = "sphinx_rtd_theme"

#
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "collapse_navigation": True,
    "navigation_depth": 3,
}

#
# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None
# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None
# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

#
html_logo = "_static/scpy.png"
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.

#
html_favicon = "_static/scpy.ico"

#
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "theme_override.css",
]
#
# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

#
# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

#
# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

#
# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

#
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""
#
# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

#
# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

#
# Output file base name for HTML help builder.
htmlhelp_basename = "spectrochempydoc"

#
trim_doctests_flags = True

#
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

#
html_context = {
    "current_version": "latest" if ("dev" in version) else "stable",
    "release": release,
    "base_url": "https://www.spectrochempy.fr",
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

#
# -- Options for LaTeX output ----------------------------------------------------------

#
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass  [
# howto/manual]).
latex_documents = [
    (
        "index",
        "spectrochempy.tex",
        "SpectroChemPy Documentation",
        "A. Travert & C. Fernandez",
        "manual",
        False,
    ),
]

#
# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "_static/scpy.png"

#
# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

#
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",  # ''letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # remove blank pages (between the title page and the TOC, etc.)
    "classoptions": ",openany,oneside",
    "babel": "\\usepackage[english]{babel}",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
\usepackage{hyperref}
\setcounter{tocdepth}{3}
""",
}

#
# If false, no module index is generated.
latex_use_modindex = True

#
# If true, show page references after internal links.
# latex_show_pagerefs = False

#
# If true, show URL addresses after external links.
# latex_show_urls = False

#
# Documents to append as an appendix to all manuals.
# latex_appendices = []

#
# If false, no module index is generated.
# latex_domain_indices = True


#
# -- Options for PDF output ---------------------------------------

#
# Grouping the document tree into PDF files. List of tuples
# (source start file, target name, title, author).
pdf_documents = [
    (
        "index",
        "SpectroChempy",
        "Spectrochempy Documentation",
        "A. Travert & C. Fernandez",
    ),
]

#
# A comma-separated list of custom stylesheets. Example:
pdf_stylesheets = ["sphinx", "kerning", "a4"]

#
# Create a compressed PDF
# Use True/False or 1/0
# Example: compressed=True
# pdf_compressed=False

#
# A colon-separated list of folders to search for fonts. Example:
# pdf_font_path=['/usr/share/fonts', '/usr/share/texmf-dist/fonts/']

#
# Language to be used for hyphenation support
pdf_language = "en_EN"

#
# If false, no index is generated.
# pdf_use_index = True

#
# If false, no modindex is generated.
# pdf_use_modindex = True

#
# If false, no coverpage is generated.
# pdf_use_coverpage = True


#
# Sphinx-gallery ---------------------------------------------------------------

#
# Generate the plots for the gallery

#
sphinx_gallery_conf = {
    "plot_gallery": "True",
    "backreferences_dir": "gettingstarted/gallery/backreferences",
    "doc_module": ("spectrochempy",),
    "reference_url": {
        "spectrochempy": None,
    },
    # path to the examples scripts
    "examples_dirs": "gettingstarted/examples",
    # path where to save gallery generated examples=======
    "gallery_dirs": "gettingstarted/gallery/auto_examples",
    "abort_on_example_error": False,
    "expected_failing_examples": [],
    "download_all_examples": False,
}
suppress_warnings = [
    "sphinx_gallery",
]

# nbsphinx ---------------------------------------------------------------------

#
# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'jpg', 'png'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

#
# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = "always"
nbsphinx_allow_errors = True
nbsphinx_timeout = 90
nbsphinx_prolog = """
"""
nbsphinx_epilog = """
"""
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

#
# Use this kernel instead of the one stored in the notebook metadata:
nbsphinx_kernel_name = "python3"

# Support for notebook formats other than .ipynb
nbsphinx_custom_formats = {
    # ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
    # ".md": ["jupytext.reads", {"fmt": "Rmd"}],
}


# configuration for intersphinx ------------------------------------------------

#
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "SciPy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


#
# linkcode ---------------------------------------------------------------------

#
def linkcode_resolve(domain, info):
    # Resolve function for the linkcode extension.
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(scp.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        fs = find_source()
        filename = "spectrochempy/%s#L%d-L%d" % fs
    except TypeError:
        return None
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "master"
    return f"https://github.com/spectrochempy/spectrochempy/blob/{tag}/{filename}"


#
# Autosummary ------------------------------------------------------------------

#
autosummary_generate = True

#
autodoc_typehints = "none"

#
napoleon_use_param = False
napoleon_use_rtype = False

#
numpydoc_class_members_toctree = True
numpydoc_show_class_members = False
numpydoc_use_plots = True

#
autoclass_content = "both"
# Both the class’ and the __init__ method’s docstring are concatenated and inserted.

#
autodoc_default_flags = ["autosummary"]

#
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


#
def autodoc_skip_member(app, what, name, obj, skip, options):
    doc = True if obj.__doc__ is not None else False
    exclude = name in exclusions or "trait" in name or name.startswith("_") or not doc
    return skip or exclude


def shorter_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Prevent displaying self in signature.
    """
    if what == "data":
        signature = "(dataset)"
        what = "function"

    if what not in ("function", "method", "data") or signature is None:
        return  # removed"class",

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
            print(sig_obj)

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
    app.add_css_file("theme_override.css")  # also can be a full URL
    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)
