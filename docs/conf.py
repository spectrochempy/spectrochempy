# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
SpectroChemPy documentation build configuration file

"""

import sys, os
import sphinx_rtd_theme # Theme for the website
import warnings

import spectrochempy

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

# sys.path.append(os.path.abspath('../sphinxext'))

# hack to make import
sys._called_from_sphinx = True

extensions = \
    [
        'nbsphinx',
        'sphinx.ext.mathjax',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.doctest',
        'sphinx.ext.intersphinx',
        'sphinx.ext.viewcode',
        'sphinx.ext.todo',
        'sphinx_gallery.gen_gallery',
        'spectrochempy.sphinxext.traitlets_sphinxdoc',
        'spectrochempy.sphinxext.autodocsumm',
        'matplotlib.sphinxext.plot_directive',
        'IPython.sphinxext.ipython_console_highlighting',
        'IPython.sphinxext.ipython_directive',
        'numpydoc',
    ]

# Numpy autodoc attributes
numpydoc_show_class_members = False
numpydoc_use_plots = True
numpydoc_class_members_toctree = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = spectrochempy.application.__version__ #.split('+')[0]
release = version.split('+')[0]
project = f"SpectroChemPy v{version}"
copyright = spectrochempy.application.__copyright__


# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

exclude_patterns = []
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')
exclude_patterns.append('_static')
exclude_patterns.append('**.ipynb_checkpoints')
exclude_patterns.append('gen_modules')
exclude_patterns.append('~temp')

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'obj'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.

rst_epilog = """

.. |ndarray| replace:: :class:`~numpy.ndarray`

.. |ma.ndarray| replace:: :class:`~numpy.ma.array`

.. |scpy| replace:: **SpectroChemPy**

.. |Project| replace:: :class:`~spectrochempy.core.projects.project.Project`

.. |Script| replace:: :class:`~spectrochempy.core.dataset.scripts.Script`

.. |NDArray| replace:: :class:`~spectrochempy.core.dataset.ndarray.NDArray`

.. |NDDataset| replace:: :class:`~spectrochempy.core.dataset.nddataset.NDDataset`

.. |NDPanel| replace:: :class:`~spectrochempy.core.dataset.ndpanel.NDPanel`

.. |Coord| replace:: :class:`~spectrochempy.core.dataset.ndcoord.Coord`

.. |CoordRange| replace:: :class:`~spectrochempy.core.dataset.ndcoordrange.CoordRange`

.. |CoordSet| replace:: :class:`~spectrochempy.core.dataset.ndcoordset.CoordSet`

.. |NDIO| replace:: :class:`~spectrochempy.core.dataset.ndio.NDIO`

.. |NDMath| replace:: :class:`~spectrochempy.core.dataset.ndmath.NDMath`

.. |Meta| replace:: :class:`~spectrochempy.core.dataset.ndmeta.Meta`

.. |NDPlot| replace:: :class:`~spectrochempy.core.dataset.ndplot.NDPlot`

.. |Unit| replace:: :class:`~spectrochempy.units.units.Unit`

.. |Quantity| replace:: :class:`~spectrochempy.units.units.Quantity`

.. |Measurement| replace:: :class:`~spectrochempy.units.units.Measurement`

.. |Axes| replace:: :class:`~matplotlib.Axes`

.. |userguide| replace:: :ref:`userguide`

"""

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'collapse_navigation': True,
    'navigation_depth': 2,
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/scpy.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/scpy.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'spectrochempydoc'


# SINCE the new sphinx version HTML5 is generated by default,
# see https://github.com/sphinx-doc/sphinx/issues/6472
# TODO: watches future change with numpydoc
html4_writer = True

trim_doctests_flags = True

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

html_context = {
    'current_version': 'dev' if 'dev' in version else 'stable',
    'release': spectrochempy.application.__release__,
    'versions': (
        ('dev', '/dev/index.html"'),
        ('stable', '/stable/index.html'),
    )
}

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass  [
# howto/manual]).
latex_documents = [(
'index', 'spectrochempy.tex', u'SpectroChemPy Documentation', u'A. Travert & C. Fernandez',
'manual', False),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "_static/scpy.png"

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'a4paper',  # ''letterpaper',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '10pt',

# remove blank pages (between the title page and the TOC, etc.)
'classoptions': ',openany,oneside',
'babel' : '\\usepackage[english]{babel}',

# Additional stuff for the LaTeX preamble.
'preamble': r'''
\usepackage{hyperref}
\setcounter{tocdepth}{3}
'''
}

# If false, no module index is generated.
latex_use_modindex = True

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for PDF output ---------------------------------------

# Grouping the document tree into PDF files. List of tuples
# (source start file, target name, title, author).
pdf_documents = [
    ('index', u'SpectroChempy', u'Spectrochempy Documentation', u'A. Travert & C. Fernandez'),
]

# A comma-separated list of custom stylesheets. Example:
pdf_stylesheets = ['sphinx','kerning','a4']

# Create a compressed PDF
# Use True/False or 1/0
# Example: compressed=True
#pdf_compressed=False

# A colon-separated list of folders to search for fonts. Example:
# pdf_font_path=['/usr/share/fonts', '/usr/share/texmf-dist/fonts/']

# Language to be used for hyphenation support
pdf_language="en_EN"

# If false, no index is generated.
#pdf_use_index = True

# If false, no modindex is generated.
#pdf_use_modindex = True

# If false, no coverpage is generated.
#pdf_use_coverpage = True

# Autosummary ------------------------------------------------------------------

autosummary_generate = True

autoclass_content = 'both'  # Both the class’ and the __init__ method’s
# docstring are concatenated and inserted.

autodoc_default_flags = ['autosummary']

exclusions = (
    'with_traceback', 'with_traceback', 'observe', 'unobserve', 'observe',
    'cross_validation_lock', 'unobserve_all', 'class_config_rst_doc',
    'class_config_section', 'class_get_help', 'class_print_help',
    'section_names', 'update_config', 'clear_instance',
    'document_config_options', 'flatten_flags', 'generate_config_file',
    'initialize_subcommand', 'initialized', 'instance',
    'json_config_loader_class', 'launch_instance', 'setup_instance',
    'load_config_file',
    'parse_command_line', 'print_alias_help', 'print_description',
    'print_examples', 'print_flag_help', 'print_help', 'print_subcommands',
    'print_version', 'python_config_loader_class', 'raises',)


def autodoc_skip_member(app, what, name, obj, skip, options):
    exclude = name in exclusions or 'trait' in name
    return skip or exclude


def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
    app.add_stylesheet("theme_override.css")  # also can be a full URL
    # app.add_stylesheet("ANOTHER.css")
    # app.add_stylesheet("AND_ANOTHER.css")


# Sphinx-gallery ---------------------------------------------------------------

# Generate the plots for the gallery

sphinx_gallery_conf = {
    'plot_gallery': 'True',
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('spectrochempy', ),
                   'reference_url': {
                        'spectrochempy': None,
                        },
    # path to the examples scripts
    'examples_dirs': 'user/examples',
    # path where to save gallery generated examples=======
    'gallery_dirs': 'gallery/auto_examples',
    'abort_on_example_error': False,
    'expected_failing_examples': [],
    'download_all_examples': False,
    }

# nbsphinx ---------------------------------------------------------------------

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'jpg', 'png'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = 'always'
nbsphinx_allow_errors = True
nbsphinx_timeout = 180

# Use this kernel instead of the one stored in the notebook metadata:
nbsphinx_kernel_name = 'python3'

# set a filename by default for notebook which have file dialogs
os.environ['TUTORIAL_FILENAME']='wodger.spg'

# set a flag to deactivate TQDM
os.environ['USE_TQDM']='No'

# configuration for intersphinx ------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7/', None),
    'pytest': ('https://docs.pytest.org/en/latest/', None),
    'ipython': ('https://ipython.readthedocs.io/en/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# linkcode ---------------------------------------------------------------------

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return \
    "https://bitbucket.org/spectrocat/spectrochempy/src/spectrochempy/%s.py" \
    % filename


