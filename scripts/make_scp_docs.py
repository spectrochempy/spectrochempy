# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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



"""Clean, build, and release the HTML documentation for spectrochempy.
"""

import os, sys
import re
import shutil
from collections import namedtuple
from glob import glob
from pkgutil import walk_packages
from subprocess import call, getoutput
from warnings import warn

from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.errors import SphinxError
from sphinx.application import Sphinx

# set the correct backend for sphinx-gallery
import matplotlib as mpl
mpl.use('agg')

from spectrochempy.api import scp

import logging
log = logging.getLogger(__name__)

scp.log_level = logging.INFO

#from sphinx.util.console import bold, darkgreen
#TODO: make our message colored too!   look at https://github.com/sphinx-doc/sphinx/blob/master/tests/test_util_logging.py
#from sphinx.cmdline import main as sphinx_build

SERVER = os.environ.get('SERVER_FOR_LCS', None)

DOCDIR = os.path.join(\
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

SOURCE = os.path.join(DOCDIR, 'source')
BUILDDIR = os.path.join(DOCDIR, '..', '..','spectrochempy_doc')
DOCTREES = os.path.join(DOCDIR, '..', '..','spectrochempy_doc', '~doctrees')

def make_docs(*options):
    """Make the html and pdf documentation

    """
    options = list(options)

    DEBUG = 'DEBUG' in options

    if DEBUG:
        scp.log_level = logging.DEBUG

    builder=None

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

        write_download_page()

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
            res = call([cmd], shell=True, executable='/bin/bash')
            log.info(res)

        log.info(
        u"\n\nBuild finished. The {0} pages are in {1}/www/{2}.".format(
            builder.upper(), BUILDDIR, builder))

    if 'release' in options:
        release()


def release():
    """Release/publish the documentation to the webpage.
    """

    # make the doc
    # make_docs(*args)

    # upload docs to the remote web server
    if SERVER:

        log.info(u"uploads to the server of the html/pdf files")
        cmd = 'rsync -e ssh -avz  --exclude="~*"    ' \
              '../../spectrochempy_doc/*   '+SERVER+':spectrochempy/'
        print(cmd)
        res = call([cmd], shell=True, executable='/bin/bash')
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


def list_packages(package):
    """Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709

    names = [package.__name__]
    for __, name, __ in walk_packages(package.__path__,
                                      prefix=package.__name__ + '.',
                                      onerror=lambda x: None):
        names.append(name)

    return names


TEMPLATE = """{headerline}

:mod:`{package}`
{underline}

.. automodule:: {package}

{subpackages}

"""  # This is important to align the text on the border


def update_rest():
    """Update the Sphinx ReST files for the package .
    """

    # Remove the existing files.
    files = glob(os.path.join(DOCDIR, 'source', 'dev', 'generated', 'auto_examples',
                              'spectrochempy*.rst'))
    for f in files:
        os.remove(f)

    # Create new files.
    import spectrochempy
    packages = list_packages(spectrochempy)

    pack = packages[:]  # remove extern librairies
    for package in pack:
        if 'extern' in package:
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
            subpackages = ".. toctree::\n\t:maxdepth: 1\n"
            for sub in sorted(subs):
                subpackages += "\n\t{0}".format(package + '.' + sub)

        # temporary import as to check the presence of doc functions
        title = "_".join(package.split('.')[1:])
        headerline = ".. _mod_{}:".format(title)

        with open(os.path.join(DOCDIR, "source", "dev", "generated",
                               "%s.rst" % package), 'w') as f:
            f.write(TEMPLATE.format(package=package,
                                    headerline=headerline,
                                    underline='=' * (len(package) + 7),
                                    subpackages=subpackages))


def write_download_page():
    """
    Modify the download item of the sidebars

    Returns
    -------

    """
    date_release = getoutput("git log -1 --tags --date='short' --format='%ad'")
    date_version = getoutput("git log -1 --date='short' --format='%ad'")

    rpls = """<h3>Download</h3>
    <ul>
      <li><a itemprop="downloadUrl"
             href="https://bitbucket.org/spectrocat/spectrochempy/get/{0}.zip"
             rel="nofollow">Latest release <br>{0} ({1})</a>
      </li>
      <li><a itemprop="downloadUrl"
             href="https://bitbucket.org/spectrocat/spectrochempy/get/master.zip"
             rel="nofollow">Latest development version <br>{2} ({3})</a>
      </li>
    </ul>

    """.format(scp.release, date_release, scp.version, date_version)

    with open(os.path.join(DOCDIR, 'source', '_templates', 'download.html'),
              "w") as f:
        f.write(rpls)



if __name__ == '__main__':

    if len(sys.argv) < 2:
        # full make
        sys.argv.append('clean')
        sys.argv.append('html')

    action = sys.argv[1]

    make_docs(*sys.argv[1:])

