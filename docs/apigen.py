# -*- coding: utf-8 -*-
"""
    sphinx.apidoc
    ~~~~~~~~~~~~~

    Parses a directory tree looking for Python modules and packages and creates
    ReST files appropriately to create code documentation with Sphinx.  It also
    creates a modules index (named modules.<suffix>).

    This is derived from the "sphinx-autopackage" script, which is:
    Copyright 2008 Société des arts technologiques (SAT),
    http://www.sat.qc.ca/

    :copyright: Copyright 2007-2017 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE_SPHINX for details.

"""

# the API

import os
import sys
import shutil
from fnmatch import fnmatch

from sphinx.util import rst
from sphinx.util.osutil import FileAvoidWrite, walk
from spectrochempy.sphinxext.traitlets_sphinxdoc import reverse_aliases, class_config_rst_doc

from traitlets import import_item
import inspect

# automodule options
OPTIONS = ['no-members', 'no-inherited-members', ]

INITPY = '__init__.py'
PY_SUFFIXES = {'.py', '.pyx'}


def makename(package, module):
    """Join package and module with a dot."""

    # Both package and module can be None/empty.
    if package:
        name = package
        if module:
            name += '.' + module
    else:
        name = module
    return name


def write_file(name, text, opts):
    """Write the output file for module/package <name>."""
    if name == 'spectrochempy':
        return
    fname = os.path.join(opts.destdir, '%s.%s' % (name, opts.suffix))
    if opts.dryrun:
        print('Would create file %s.' % fname)
        return
    if not opts.force and os.path.isfile(fname):
        print('File %s already exists, skipping.' % fname)
    else:
        print('Creating file %s.' % fname)
        with FileAvoidWrite(fname) as f:
            f.write(text)


def format_heading(level, text, escape=True):
    """Create a heading of <level> [1, 2 or 3 supported]."""
    if escape:
        text = rst.escape(text)
    underlining = ['=', '-', '~', ][level - 1] * len(text)
    return '%s\n%s\n\n' % (text, underlining)


def format_directive(module, package=None, auto='automodule'):
    """Create the automodule directive and add the options."""
    item = makename(package, module)

    if auto == 'automodule':
        directive = '.. currentmodule:: %s\n\n' % item
    else:
        directive = ''
    directive += '.. %s:: %s\n' % (auto, item)
    for option in OPTIONS:
        directive += '   :%s:\n' % option
    return directive


def create_module_file(package, module, opts):
    """Build the text of the file and write the file."""
    # generate separate file for this module

    item = makename(package, module)
    text = ".. _mod_{}:\n\n".format("_".join(item.split('.')))
    if not opts.noheadings:
        text += format_heading(1, '%s module' % module)
    else:
        text += ''

    write_file(makename(package, module), text, opts)


def create_package_file(root, master_package, subroot, py_files, opts, subs,
                        is_namespace):
    """Build the text of the file and write the file."""

    name = makename(master_package, subroot)
    _name = name.replace('.', '_')
    text = ".. _api_%s:\n\n" % _name
    text += format_heading(1, (
        '%s package' if not is_namespace else "%s namespace") % name)

    if opts.modulefirst and not is_namespace:
        text += format_directive(subroot, master_package)
        text += '\n'

    # build a list of directories that are subpackages (contain an INITPY file)
    subs = [sub for sub in subs if
            os.path.isfile(os.path.join(root, sub, INITPY))]
    # if there are some package directories, add a TOC for theses subpackages
    if subs:
        text += format_heading(2, 'Subpackages')
        text += '.. autosummary::\n'
        text += '   :toctree: \n'
        text += '   :template: module.rst\n\n'
        for sub in subs:
            text += '   %s.%s\n' % (makename(master_package, subroot), sub)
        text += '\n'

    submods = [os.path.splitext(sub)[0] for sub in py_files if
               not shall_skip(os.path.join(root, sub), opts) and sub != INITPY]
    if submods:
        text += format_heading(2, 'Submodules')
        if opts.separatemodules:
            # text += '.. toctree::\n\n'
            text += '.. autosummary::\n'
            text += '   :toctree: \n'
            text += '   :template: module.rst\n\n'

            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                text += '   %s\n' % modfile
                # create_module_file(master_package,
                #                    makename(subroot, submod), opts)
        else:
            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                if not opts.noheadings:
                    text += format_heading(2, '%s module' % modfile)
                text += format_directive(makename(subroot, submod),
                                         master_package)
                text += '\n'
        text += '\n'

    if not opts.modulefirst and not is_namespace:
        text += format_heading(2, 'Module contents')
        text += format_directive(subroot, master_package)

    write_file(makename(master_package, subroot), text, opts)


def create_modules_toc_file(modules, opts, name='modules'):
    """Create the module's index."""
    text = format_heading(1, '%s' % opts.header, escape=False)
    text += '.. toctree:: \n'
    text += '   :maxdepth: %s\n\n' % opts.maxdepth

    modules.sort()
    prev_module = ''
    for module in modules:
        # look if the module is a subpackage and, if yes, ignore it
        if module.startswith(prev_module + '.'):
            continue
        prev_module = module
        text += '   %s\n' % module

    write_file(name, text, opts)


def shall_skip(module, opts):
    """Check if we want to skip this module."""
    # skip if the file doesn't exist and not using implicit namespaces
    if not opts.implicit_namespaces and not os.path.exists(module):
        return True

    # skip it if there is nothing (or just \n or \r\n) in the file
    if os.path.exists(module) and os.path.getsize(module) <= 2:
        return True

    # skip if it has a "private" name and this is selected
    filename = os.path.basename(module)
    if filename != '__init__.py' and filename.startswith(
            '_') and not opts.includeprivate:
        return True

    # skip if it follows some of the exclude pattern
    for pattern in opts.exclude_patterns:
        if fnmatch(module, "*/{}".format(pattern)):
            if not ('*' in pattern or '?' in pattern or '[' in pattern):
                # check that the file is exactly the pattern
                if os.path.basename(module) != pattern:
                    continue
            return True

    return False


def recurse_tree(rootpath, excludes, opts):
    """
    Look for every file in the directory tree and create the corresponding
    ReST files.
    """
    # check if the base directory is a package and get its name
    if INITPY in os.listdir(rootpath):
        root_package = rootpath.split(os.path.sep)[-1]
    else:
        # otherwise, the base is a directory with packages
        root_package = None

    toplevels = []
    followlinks = getattr(opts, 'followlinks', False)
    includeprivate = getattr(opts, 'includeprivate', False)
    implicit_namespaces = getattr(opts, 'implicit_namespaces', False)
    for root, subs, files in walk(rootpath, followlinks=followlinks):
        # document only Python module files (that aren't excluded)
        py_files = sorted(f for f in files if os.path.splitext(f)[
            1] in PY_SUFFIXES and not is_excluded(os.path.join(root, f),
                                                  excludes))
        is_pkg = INITPY in py_files
        is_namespace = INITPY not in py_files and implicit_namespaces
        if is_pkg:
            py_files.remove(INITPY)
            py_files.insert(0, INITPY)
        elif root != rootpath:
            # only accept non-package at toplevel unless using implicit
            # namespaces
            if not implicit_namespaces:
                del subs[:]
                continue
        # remove hidden ('.') and private ('_') directories, as well as
        # excluded dirs
        if includeprivate:
            exclude_prefixes = ('.', '__')
        else:
            exclude_prefixes = ('.', '_')
        subs[:] = sorted(sub for sub in subs if not sub.startswith(
            exclude_prefixes) and not is_excluded(os.path.join(root, sub),
                                                  excludes))

        if is_pkg or is_namespace:
            # we are in a package with something to document
            if subs or len(py_files) > 1 or not shall_skip(
                    os.path.join(root, INITPY), opts):
                subpackage = root[len(rootpath):].lstrip(os.path.sep).replace(
                    os.path.sep, '.')
                # if this is not a namespace or
                # a namespace and there is something there to document
                if not is_namespace or len(py_files) > 0:
                    create_package_file(root, root_package, subpackage,
                                        py_files, opts, subs, is_namespace)
                    toplevels.append(makename(root_package, subpackage))
        else:
            # if we are at the root level, we don't require it to be a package
            assert root == rootpath and root_package is None
            for py_file in py_files:
                if not shall_skip(os.path.join(rootpath, py_file), opts):
                    module = os.path.splitext(py_file)[0]
                    create_module_file(root_package, module, opts)
                    toplevels.append(module)

    return toplevels


def normalize_excludes(rootpath, excludes):
    """Normalize the excluded directory list."""
    return [os.path.abspath(os.path.join(rootpath, exclude)) for exclude in
            excludes]


def is_excluded(root, excludes):
    """Check if the directory is in the exclude list

    Note: by having trailing slashes, we avoid common prefix issues, like
          e.g. an exlude "foo" also accidentally excluding "foobar".
    """
    for exclude in excludes:
        if fnmatch(root, exclude):
            return True
    return False


def main(rootpath, destdir='./api/generated', exclude_dirs=[],
         **kwargs):
    """
    Modified version of apidoc

    Parameters
    ----------
    rootpath: str
        Path of the package to document. If not given, we will try to guess it
        from the location of apidoc.
    destdir: str, optional
        Path of the output file. By default output='./api/'
    exclude_dirs: list of str, optional
        directory names to exclude from parsing documentation

    Other parameters
    ----------------
    exclude_patterns: list of str, optional
        pattern for filenames to exclude
    force: bool, optional
        if False old ``rst`` file will not be overwritten
    dryrun: bool, optional
        if True, no output file will be created
    notoc: bool, optional
        No table of content (TOC)produced
    maxdepth: bool
        Maximum depth of submodules to show in the TOC
    followlinks: bool
        Follow symbolic links
    separatemodules: bool, optional
        Put documentation for each module on its own page. Default= True
    includeprivate: bool, optional
        Include ``_private`` modules. Default= `False`
    noheadings: bool
        Don't create headings for the module/package packages (e.g. when the
        docstrings already contain them
    modulefirst: bool
        Put module documentation before submodule documentation
    implicit_namespaces: bool
        Interpret module paths according to PEP-0420 implicit namespaces
        specification
    suffix: str, optioanl
        file suffix, default='rst'

    Returns
    -------
    done: bool

    """

    class Options(dict):
        def __init__(self, *args, **kwargs):
            super(Options, self).__init__(*args, **kwargs)
            self.__dict__ = self

    # default options
    opts = Options({
        'exclude_patterns': [], 'force': True, 'tocdepth': 1,
        'followlinks': False, 'dryrun': False, 'separatemodules': True,
        'includeprivate': False, 'notoc': True, 'noheadings': False,
        'modulefirst': True, 'implicit_namespaces': True, 'suffix': 'rst',
        'developper': False, 'genapi': False,
    })

    # get options form kwargs
    opts.update(kwargs)

    destdir = os.path.abspath(destdir)

    if opts.force:
        shutil.rmtree(destdir, ignore_errors=True)
    if not opts.dryrun:
        os.makedirs(destdir, exist_ok=opts.force)
    if not os.path.isdir(destdir):
        print('%s is not a directory.' % rootpath, file=sys.stderr)
        sys.exit(1)
    opts['destdir'] = destdir

    if opts.genapi:
        create_api_files(rootpath, opts)
        return

    if not os.path.exists(rootpath):
        # try to guess!
        _rootpath = rootpath
        dirname = os.path.dirname(__file__)
        while not os.path.exists(rootpath):
            rootpath = os.path.join(dirname, _rootpath)
            dirname = os.path.dirname(dirname)
            # print(rootpath)

    if not os.path.isdir(rootpath) and not opts.genapi:
        print('%s is not a directory.' % rootpath, file=sys.stderr)
        sys.exit(1)

    rootpath = os.path.abspath(rootpath)
    excludes = normalize_excludes(rootpath, exclude_dirs)
    modules = recurse_tree(rootpath, excludes, opts)

    if not opts.notoc:
        create_modules_toc_file(modules, opts)

    return True


def create_api_files(rootpath, opts):
    """Build the text of the file and write the file."""
    # generate separate file for the members of the api

    api = rootpath
    _imported_item = import_item(api)

    clsmembers = inspect.getmembers(_imported_item)

    members = [m for m in clsmembers if
               m[0] in _imported_item.__all__ and not m[0].startswith('__')]

    indextemplate = """.. _api_reference_spectrochempy:

User API reference
==================

.. currentmodule:: spectrochempy

The |scpy| API exposes many objects and functions that are described below.

To use the API, one must load it using one of the following syntax:

>>> import spectrochempy as scp

>>> from spectrochempy import *

In the second syntax, as usual in python, access to the objects/functions 
may be simplified (*e.g.*, we can use `plot_stack` instead of  `scp.plot_stack` but there is always a risk of 
overwriting some variables already in the namespace. Therefore, the first syntax is in general 
recommended,
although that, in the examples in this documentation, we have often use the 
second one for simplicity.

To go deeper in the core of |scpy|, look at 
the :ref:`Developper's documentation<develdocs>` and 
:ref:`Code Reference <develreference>`.

Objects
-------

.. autosummary::
   :toctree:

{classes}

Functions
---------

.. currentmodule:: spectrochempy

.. autosummary::
   :toctree:

{funcs}

Preferences
-----------

{preferences}

Constants
---------

{consts}

"""

    classtemplate = """{api}.{klass}
==============================================================================

.. automodule:: {api}

.. autoclass:: {api}.{klass}
   :members:

.. include:: /gen_modules/backreferences/{api}.{klass}.examples

.. raw:: html

   <div style='clear:both'></div>

"""

    functemplate = """{api}.{func}
==============================================================================

.. automodule:: {api}

.. autofunction:: {api}.{func}
 
.. include:: /gen_modules/backreferences/{api}.{func}.examples

.. raw:: html

   <div style='clear:both'></div>

"""

    lconsts = [":%s: %s\n" % m for m in members if
               type(m[1]) in [int, float, str, bool, tuple]]
    lclasses = []
    classes = [m[0] for m in members if
               inspect.isclass(m[1]) and not type(m[1]).__name__ == 'type']
    for klass in classes:
        name = "{api}.{klass}".format(api=api, klass=klass)
        text = classtemplate.format(api=api, klass=klass)
        write_file(name, text, opts)
        lclasses.append(name + '\n')

    lfuncs = []
    funcs = [m[0] for m in members if
             inspect.isfunction(m[1]) or inspect.ismethod(m[1])]
    for func in funcs:
        name = "{api}.{func}".format(api=api, func=func)
        text = functemplate.format(api=api, func=func)
        write_file(name, text, opts)
        lfuncs.append(name + '\n')

    _classes = "    ".join(lclasses)
    _funcs = "    ".join(lfuncs)
    _consts = "".join(lconsts)

    text = indextemplate.format(consts=_consts, preferences=write_prefs(),
                                classes="    " + _classes,
                                funcs="    " + _funcs)
    write_file('index', text, opts)


def write_prefs():
    from spectrochempy.core import app

    trait_aliases = reverse_aliases(app)
    text = ""
    for c in app._classes_inc_parents():
        text += class_config_rst_doc(c, trait_aliases)
        text += '\n'
    return text


if __name__ == "__main__":

    PROJECT = "spectrochempy"

    DOCDIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    API = os.path.join(DOCDIR, 'api', 'generated')
    DEVAPI = os.path.join(DOCDIR, 'dev', 'generated')

    main(PROJECT, tocdepth=1, includeprivate=True, destdir=DEVAPI,
                exclude_patterns=['api.py', "test_*.py", "tests"],
                exclude_dirs=['extern', 'sphinxext', '~misc', 'gui', 'tests','*/tests','*/*/tests'], )