# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

"""Clean, build, and release the HTML and PDF documentation for SpectroChemPy.

usage::

    python builddocs clean html pdf release

where optional parameters idincates which job to perfom.

"""

import os, sys
import re
import shutil
from collections import namedtuple
from glob import glob
from pkgutil import walk_packages
import subprocess
from warnings import warn
import setuptools_scm

from sphinx.util.docutils import docutils_namespace, patch_docutils
from ipython_genutils.text import indent, dedent, wrap_paragraphs
from sphinx.errors import SphinxError
from sphinx.application import Sphinx

# set the correct backend for sphinx-gallery
import matplotlib as mpl
mpl.use('agg')

from spectrochempy.api import *
from spectrochempy.utils import (list_packages, get_version,
                                 get_release_date, get_version_date)
from traitlets import import_item

import logging
log_level = logging.ERROR

#from sphinx.util.console import bold, darkgreen
#TODO: make our message colored too!   look at https://github.com/sphinx-doc/sphinx/blob/master/tests/test_util_logging.py
#from sphinx.cmdline import main as sphinx_build

SERVER = os.environ.get('SERVER_FOR_LCS', None)

DOCDIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

SOURCE = os.path.join(DOCDIR, 'source')
BUILDDIR = os.path.join(DOCDIR, '..', '..','spectrochempy_doc')
DOCTREES = os.path.join(DOCDIR, '..', '..','spectrochempy_doc', '~doctrees')

def gitcommands():

    COMMIT = False

    pipe = subprocess.Popen(
            ["git", "status"],
            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()

    if "nothing to commit" not in so.decode("ascii"):
        COMMIT = True

    if COMMIT:

        pipe = subprocess.Popen(
                ["git", "add", "-A"],
                stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()

        pipe = subprocess.Popen(
                ["git", "log", "-1", "--pretty=%B"],
                stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()
        OUTPUT = so.decode("ascii")

        pipe = subprocess.Popen(
                ["git", "commit", "--no-verify", "--amend", "-m", '%s' % OUTPUT],
                stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()

        pass

def make_docs(*options):
    """Make the html and pdf documentation

    """
    # make sure commits have been done
    gitcommands()

    options = list(options)

    DEBUG = 'DEBUG' in options

    if DEBUG:
        log_level = logging.DEBUG

    builders = []
    if  'html' in options:
        builders.append('html')

    if 'pdf' in options:
        builders.append('latex')

    if 'clean' in options:
        clean()
        make_dirs()
        options.remove('clean')
        log.info('\n\nOld documentation now erased.\n\n')

    for builder in builders:
        srcdir = confdir = SOURCE
        outdir = "{0}/{1}".format(BUILDDIR, builder)
        doctreedir = "{0}/~doctrees".format(BUILDDIR)

        #with patch_docutils(), docutils_namespace():
        sp = Sphinx(srcdir, confdir, outdir, doctreedir, builder)
        sp.verbosity = 0

        update_rest()

        sp.build()
        res = sp.statuscode
        log.debug(res)

        if builder=='latex':
            cmd = "cd {}/latex; " \
              "make; mv spectrochempy.pdf " \
              " ../pdf/spectrochempy.pdf".format(BUILDDIR)
            res = subprocess.call([cmd], shell=True, executable='/bin/bash')
            log.info(res)

        gitcommands()  # update repository

        log.info(
        "\n\nBuild finished. The {0} pages are in {1}/www/{2}.".format(
            builder.upper(), BUILDDIR, builder))

    if 'release' in options:
        do_release()


def do_release():
    """Release/publish the documentation to the webpage.
    """

    # make the doc
    # make_docs(*args)

    # upload docs to the remote web server
    if SERVER:

        log.info("uploads to the server of the html/pdf files")
        cmd = 'rsync -e ssh -avz  --exclude="~*"    ' \
              '../spectrochempy_doc/*   '+SERVER+':spectrochempy/'
        print(subprocess.call(['pwd'], shell=True, executable='/bin/bash'))
        print(cmd)
        res = subprocess.call([cmd], shell=True, executable='/bin/bash')
        log.info(res)
        log.info('\n'+cmd + "Finished")

    else:
        log.error ('Cannot find the upload server: {}!'.format(SERVER))


def clean():
    """Clean/remove the built documentation.
    """

    shutil.rmtree(BUILDDIR + '/html', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/pdf', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/latex', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/~doctrees', ignore_errors=True)
    shutil.rmtree(SOURCE + '/api/auto_examples', ignore_errors=True)
    shutil.rmtree(SOURCE + '/gen_modules', ignore_errors=True)
    shutil.rmtree(SOURCE + '/api/auto_examples', ignore_errors=True)
    shutil.rmtree(SOURCE + '/modules/generated', ignore_errors=True)
    shutil.rmtree(SOURCE + '/dev/generated', ignore_errors=True)


def make_dirs():
    """Create the directories required to build the documentation.
    """

    # Create regular directories.
    build_dirs = [os.path.join(BUILDDIR, '~doctrees'),
                  os.path.join(BUILDDIR, 'html'),
                  os.path.join(BUILDDIR, 'latex'),
                  os.path.join(BUILDDIR, 'pdf'),
                  os.path.join(SOURCE, '_static'),
                  os.path.join(SOURCE, 'dev', 'generated')
                  ]
    for d in build_dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass


TEMPLATE = """{headerline}

:mod:`{package}`
{underline}


.. automodule:: {package}

{methods}

{classes}    
   
{subpackages}

"""  # This is important to align the text on the border

#**Inheritance diagram**:

#.. inheritance-diagram:: {package}
#   :parts: 3

def class_config_rst_doc(cls):
    """Generate rST documentation for the class `cls` config options.
    Excludes traits defined on parent classes. (adapted from traitlets)
    """
    lines = []
    for k, trait in sorted(cls.class_traits().items()):
        if trait.name.startswith('_') or not trait.help or trait.name in [
            'cli_config',
        ]:
            continue

        ttype = '`'+trait.__class__.__name__+'`'

        termline = '**'+trait.name+'**'

        # Choices or type
        if 'Enum' in ttype:
            # include Enum choices
            termline += ' : ' + '|'.join('`'+repr(x)+'`' for x in trait.values)
        else:
            termline += ' : ' + ttype + ', '
        lines.append(termline)

        # Default value
        try:
            dvr = trait.default_value_repr()
        except Exception:
            dvr = None  # ignore defaults we can't construct
        if dvr is not None:
            if len(dvr) > 64:
                dvr = dvr[:61] + '...'
            # Double up backslashes, so they get to the rendered docs
            dvr = dvr.replace('\\n', '\\\\n')
            lines.append(indent('Default: `%s`,' % dvr, 4))
            lines.append('')

        help = trait.help or 'No description'
        lines.append(indent(dedent(help+'.'), 4))

        # Blank line
        lines.append('')

    return '\n'.join(lines)

def update_rest():
    """Update the Sphinx ReST files for the package .
    """

    # Remove the existing files.
    files = glob(os.path.join(DOCDIR, 'source', 'dev', 'generated', 'auto_examples',
                              'spectrochempy*.rst'))
    for f in files:
        os.remove(f)

    # Create new files.
    packages = []
    import spectrochempy
    packages += list_packages(spectrochempy)

    pack = packages[:]  # remove api librairies
    for package in pack:
        if 'api' in package or 'extern' in package:
            packages.remove(package)

    for package in packages:
        subs = set()
        for subpackage in packages:
            if subpackage.startswith(package) and \
                            subpackage != package:
                sub = subpackage.split(package)[1]
                if sub.startswith('.'):
                    sub = sub.split('.')[1]
                    subs.add(sub)

        if not subs:
            subpackages = ""
        else:
            subpackages = """
            
Sub-Packages
-------------

The following sub-packages are available in this package:
 
"""
            subpackages += ".. toctree::\n\t:maxdepth: 1\n"
            for sub in sorted(subs):
                if 'api' in sub:
                    continue
                subpackages += "\n\t{0}".format(package + '.' + sub)

        # temporary import as to check the presence of doc functions
        pkg = import_item(package)

        classes = ''
        methods = ''

        if hasattr(pkg, '_classes'):
            classes += "\nClasses\n-------------\n"
            classes += "This module contains the following classes:\n\n"
            for item in pkg._classes:
                _item = "%s.%s"%(package,item)
                _imported_item = import_item(_item)
                if hasattr(_imported_item,'class_config_rst_doc'):
                    doc = "\n"+class_config_rst_doc(_imported_item)
                    doc = doc.replace(item+".",'')
                    doc = doc.replace(item + "\n", '\n\t')
                    _imported_item.__doc__ = _imported_item.__doc__.format(attributes=doc) # "\n\tAttributes\n\t========================\n%s\n"%doc
                classes += "\n.. autoclass:: %s\n\t:members:\n\t:inherited-members:\n\n" % _item

        if hasattr(pkg, '_methods'):
            methods += "\nMethods\n---------------\n"
            methods += "This module contains the following methods:\n\n"

            for item in pkg._methods:
                # check if it is really a method:
                #if hasattr(getattr(spectrochempy.api,
                #                   '{}'.format(item)), '__call__'):
                _item = "%s.%s"%(package,item)
                methods += "\n.. automethod:: %s\n\n" % _item
                #else:
                #    print(item)
                #    # may be add this in the doc to


        title = "_".join(package.split('.')[1:])
        headerline = ".. _mod_{}:".format(title)

        with open(os.path.join(DOCDIR, "source", "dev", "generated",
                               "%s.rst" % package), 'w') as f:
            f.write(TEMPLATE.format(package=package,
                                    headerline=headerline,
                                    underline='#' * (len(package) + 7),
                                    classes = classes,
                                    methods=methods,
                                    subpackages=subpackages,
                                    ))


if __name__ == '__main__':

    if len(sys.argv) < 2:
        # full make
        sys.argv.append('clean')
        sys.argv.append('html')
        sys.argv.append('pdf')
        sys.argv.append('release')

    action = sys.argv[1]

    make_docs(*sys.argv[1:])

