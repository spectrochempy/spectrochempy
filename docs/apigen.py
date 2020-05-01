# -*- coding: utf-8 -*-
"""
    sphinx.apidoc (https://github.com/sphinx-doc/sphinx/blob/master/sphinx/ext/apidoc.py)
    

    Parses a directory tree looking for Python modules and packages and creates
    ReST files appropriately to create code documentation with Sphinx.  It also
    creates a modules index (named modules.<suffix>).

    This is derived from the "sphinx-autopackage" script, which is :
    Copyright 2008 Société des arts technologiques (SAT),
    http://www.sat.qc.ca/

    :copyright: Copyright 2007-2017 by the Sphinx team, see AUTHORS .
    :license: BSD, see LICENSE_SPHINX for details.

"""

# the API

import os
import shutil

from sphinx.util.osutil import FileAvoidWrite
from spectrochempy.sphinxext.traitlets_sphinxdoc import reverse_aliases, class_config_rst_doc

from traitlets import import_item
import inspect

import textwrap


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
    fname = os.path.join(opts.destdir, '%s.rst' % (name))
    if opts.dryrun:
        print('Would create file %s.' % fname)
        return
    
    with FileAvoidWrite(fname) as f:
        f.write(text)
        print('Writing file %s.' % fname)



def main(rootpath, **kwargs):
    """
    Modified version of apidoc

    Parameters
    ----------
    rootpath : str
        Path of the package to document. If not given, we will try to guess it
        from the location of apidoc.
    destdir : str, optional
        Path of the output file. By default output='./api/'

    Other parameters
    ----------------
    exclude_patterns : list of str, optional
        pattern for filenames to exclude
    force : bool, optional
        if False old ``rst`` file will not be overwritten
    dryrun : bool, optional
        if True, no output file will be created

    Returns
    -------
    done : bool

    """

    class Options(dict):
        def __init__(self, *args, **kwargs):
            super(Options, self).__init__(*args, **kwargs)
            self.__dict__ = self

    # default options
    opts = Options({
        'destdir':None,
        'exclude_patterns': [],
        'force': False,
        'dryrun': False,
    })

    # get options form kwargs
    opts.update(kwargs)

    destdir = os.path.abspath(opts.destdir)

    if opts.force:
        shutil.rmtree(destdir, ignore_errors=True)
        
    if not opts.dryrun or opts.force:
        os.makedirs(destdir, exist_ok=True)
        
    create_api_files(rootpath, opts)
    
    return


def create_api_files(rootpath, opts):
    """Build the text of the file and write the file."""
    # generate separate file for the members of the api

    project = os.path.basename(rootpath)
    _imported_item = import_item(project)

    clsmembers = inspect.getmembers(_imported_item)

    members = [m for m in clsmembers if
               m[0] in _imported_item.__all__ and not m[0].startswith('__')]

    indextemplate = textwrap.dedent("""
    .. _api_reference_spectrochempy:

    User API reference
    ==================
    
    .. currentmodule:: spectrochempy
    
    The |scpy| API exposes many objects and functions that are described below.
    
    To use the API, one must load it using one of the following syntax :
    
    >>> import spectrochempy as scp
    
    >>> from spectrochempy import *
    
    In the second syntax, as usual in python, access to the objects/functions
    may be simplified (*e.g.*, we can use `plot_stack` instead of  `scp.plot_stack` but there is always a risk of
    overwriting some variables already in the namespace. Therefore, the first syntax is in general
    recommended,
    although that, for the examples in this documentation, we have often use the
    second one for simplicity.
    
    
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
    
    """)

    classtemplate = textwrap.dedent("""
    
    {project}.{klass}
    ==============================================================================

    .. automodule:: {project}
    
    .. autoclass:: {project}.{klass}
       :members:
       :inherited-members:
    
    .. {include} /gen_modules/backreferences/{project}.{klass}.examples
    
    .. raw:: html
    
       <div style='clear:both'></div>
    
    """)
    
    functemplate = textwrap.dedent("""
    
    {project}.{func}
    ==============================================================================
    
    .. automodule:: {project}
    
    .. autofunction:: {project}.{func}
    
    .. {include} /gen_modules/backreferences/{project}.{func}.examples
    
    .. raw:: html
    
       <div style='clear:both'></div>
    
    """)

    lconsts = [":%s: %s\n" % m for m in members if
               type(m[1]) in [int, float, str, bool, tuple]]
    lclasses = []
    classes = [m[0] for m in members if
               inspect.isclass(m[1]) and not type(m[1]).__name__ == 'type']
    for klass in classes:
        if klass not in opts.exclude_patterns:
            name = "{project}.{klass}".format(project=project, klass=klass)
            example_exists = os.path.exists(f"{rootpath}/../docs/gen_modules/backreferences/{name}.examples")
            include = "include::" if example_exists else ''
            text = classtemplate.format(project=project, klass=klass, include=include)
            write_file(name, text, opts)
            lclasses.append(name + '\n')

    lfuncs = []
    funcs = [m[0] for m in members if
             inspect.isfunction(m[1]) or inspect.ismethod(m[1])]
    for func in funcs:
        name = "{project}.{func}".format(project=project, func=func)
        example_exists = os.path.exists(f"{rootpath}/../docs/gen_modules/backreferences/{name}.examples")
        include = "include::" if example_exists else ''
        text = functemplate.format(project=project, func=func, include=include)
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
    PROJECTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SOURCESDIR = os.path.join(PROJECTDIR, "spectrochempy")
    DOCDIR = os.path.join(PROJECTDIR, "docs")
    API = os.path.join(DOCDIR, 'api','generated')

    main(SOURCESDIR,
            tocdepth=1,
            force=False,
            includeprivate=True,
            destdir=API,
            exclude_patterns=[
                'NDArray',
                'NDComplexArray',
                'NDIO',
                'NDPlot',
            ],
         )