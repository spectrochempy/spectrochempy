#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

#
"""
Clean, build, and release the HTML and PDF documentation for SpectroChemPy.

usage::

    python make.py [options]

where optional parameters indicates which job(s) is(are) to perform.

"""

import argparse
import inspect
import os
import shutil
import sys
import textwrap
import warnings
import zipfile
from glob import iglob

import json5 as json
import numpy as np
import requests
from jinja2 import Template
from skimage.io import imread, imsave
from skimage.transform import resize
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx40Warning  # , RemovedInSphinx30Warning
from sphinx.util.osutil import FileAvoidWrite
from traitlets import import_item

from spectrochempy import version
from spectrochempy.utils import sh

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)
# warnings.filterwarnings(action='error')
warnings.filterwarnings(action='ignore', category=RemovedInSphinx50Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)
# warnings.filterwarnings(action='ignore', category=RemovedInSphinx30Warning)

# CONSTANT
PROJECT = "spectrochempy"
REPO_URI = f"spectrochempy/{PROJECT}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "spectrochempy.github.io/spectrochempy"

# PATHS
HOME = os.environ.get('HOME', os.path.expanduser('~'))
DOCDIR = os.path.dirname(os.path.abspath(__file__))
PROJECTDIR = os.path.dirname(DOCDIR)
SOURCESDIR = os.path.join(PROJECTDIR, PROJECT)
USERDIR = os.path.join(DOCDIR, "user")
TEMPLATES = os.path.join(DOCDIR, '_templates')
TUTORIALS = os.path.join(USERDIR, "tutorials", "*", "*.py")
USERGUIDE = os.path.join(USERDIR, "userguide", "*", "*.py")
API = os.path.join(DOCDIR, 'api', 'generated')
GALLERYDIR = os.path.join(DOCDIR, "gallery")
DOCREPO = os.path.join(HOME, "spectrochempy_docs")
DOCTREES = os.path.join(DOCREPO, "~doctrees")
HTML = os.path.join(DOCREPO, "html")
LATEX = os.path.join(DOCREPO, 'latex')
DOWNLOADS = os.path.join(HTML, 'downloads')

__all__ = []


# ======================================================================================================================
class Options(dict):
    def __init__(self, *args, **kwargs):
        super(Options, self).__init__(*args, **kwargs)
        self.__dict__ = self


# ======================================================================================================================
class Apigen(object):
    """
    borrowed and heavily modified from :
    sphinx.apidoc (https://github.com/sphinx-doc/sphinx/blob/master/sphinx/ext/apidoc.py)


    Parses a directory tree looking for Python modules and packages and creates
    ReST files appropriately to create code documentation with Sphinx.  It also
    creates a modules index (named modules.<suffix>).

    This is derived from the "sphinx-autopackage" script, which is :
    Copyright 2008 Société des arts technologiques (SAT), http://www.sat.qc.ca/

    :copyright: Copyright 2007-2017 by the Sphinx team, see AUTHORS .
    :license: BSD, see LICENSE_SPHINX for details.

    """

    def __init__(self):

        with open(os.path.join(DOCDIR, "_templates", "class.rst")) as f:
            self.class_template = f.read()

        with open(os.path.join(DOCDIR, "_templates", "function.rst")) as f:
            self.function_template = f.read()

    @staticmethod
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

    @staticmethod
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

    def create_api_files(self, rootpath, opts):
        """Build the text of the file and write the file."""
        # generate separate file for the members of the api

        project = os.path.basename(rootpath)
        _imported_item = import_item(project)

        clsmembers = inspect.getmembers(_imported_item)

        members = [m for m in clsmembers if
                   m[0] in _imported_item.__all__ and not m[0].startswith('__')]

        classtemplate = textwrap.dedent(self.class_template)

        functemplate = textwrap.dedent(self.function_template)

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
                self.write_file(name, text, opts)
                lclasses.append(name + '\n')

        lfuncs = []
        funcs = [m[0] for m in members if
                 inspect.isfunction(m[1]) or inspect.ismethod(m[1])]
        for func in funcs:
            name = "{project}.{func}".format(project=project, func=func)
            example_exists = os.path.exists(f"{rootpath}/../docs/gen_modules/backreferences/{name}.examples")
            include = "include::" if example_exists else ''
            text = functemplate.format(project=project, func=func, include=include)
            self.write_file(name, text, opts)
            lfuncs.append(name + '\n')

        _classes = "    ".join(lclasses)
        _funcs = "    ".join(lfuncs)
        _consts = "".join(lconsts)

    # ----------------------------------------------------------------------------------------------------------------------
    def __call__(self, rootpath, **kwargs):
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

        # default options
        opts = Options({
                'destdir': None,
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

        self.create_api_files(rootpath, opts)

        return


apigen = Apigen()


# ======================================================================================================================
class Build(object):

    # ..................................................................................................................
    def __init__(self):
        # determine if we are in the developement branch (latest) or master (stable)

        if 'dev' in version:
            self._doc_version = 'latest'
        else:
            self._doc_version = 'stable'

    # ..................................................................................................................
    @property
    def doc_version(self):
        return self._doc_version

    # ..................................................................................................................
    def __call__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("-H", "--html", help="create html pages", action="store_true")
        parser.add_argument("-P", "--pdf", help="create pdf manual", action="store_true")
        parser.add_argument("-T", "--tutorials", help="zip notebook tutorials for downloads", action="store_true")
        parser.add_argument("--clean", help="clean for a full regeneration of the documentation", action="store_true")
        parser.add_argument("--sync", help="sync doc ipynb using jupytext", action="store_true")
        parser.add_argument("--delnb", help="delete all ipynb", action="store_true")
        parser.add_argument("-m", "--message", default='DOCS: updated', help='optional git commit message')
        parser.add_argument("--api", help="execute a full regeneration of the api", action="store_true")
        parser.add_argument("-R", "--release", help="release the current version documentation on website",
                            action="store_true")
        parser.add_argument("-C", "--changelogs", help="update changelogs using the github issues",
                            default="latest")
        parser.add_argument("--all", help="Build all docs", action="store_true")

        args = parser.parse_args()

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            return

        self.regenerate_api = args.api

        if args.sync:
            self.sync_notebooks()
        if args.changelogs:
            self.make_changelog(args.changelogs)
        if args.delnb:
            self.delnb()
        if args.clean and args.html:
            self.clean('html')
        if args.clean and args.pdf:
            self.clean('latex')
        if args.html:
            self.make_docs('html')
        if args.pdf:
            self.make_docs('latex')
            self.make_pdf()
        if args.tutorials:
            self.make_tutorials()
        if args.all:
            self.clean('html')
            self.clean('latex')
            self.make_changelog()
            self.sync_notebooks()
            self.make_docs('html')
            self.make_docs('latex')
            self.make_pdf()
            self.make_tutorials()

    @staticmethod
    def _confirm(action):
        # private method to ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(f"OK to continue `{action}` Y[es]/[N[o] ? ", ).lower()
        return answer[:1] == "y"

    # ..................................................................................................................
    def make_docs(self, builder='html', clean=False):
        # Make the html or latex documentation

        doc_version = self.doc_version

        if builder not in ['html', 'latex']:
            raise ValueError('Not a supported builder: Must be "html" or "latex"')

        BUILDDIR = os.path.join(DOCREPO, builder)
        print(f'building {builder.upper()} documentation ({doc_version.capitalize()} version : {version})')

        # recreate dir if needed
        if clean:
            self.clean(builder)
        self.make_dirs()

        # regenate api documentation
        if (self.regenerate_api or not os.path.exists(API)):
            self.api_gen()

        # run sphinx
        srcdir = confdir = DOCDIR
        outdir = f"{BUILDDIR}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"
        sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
        sp.verbosity = 1
        sp.build()

        print(f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
              f"are in {os.path.normpath(outdir)}.")

        # do some cleaning
        shutil.rmtree(os.path.join('docs', 'auto_examples'), ignore_errors=True)

        if builder == 'html':
            self.make_redirection_page()
            self.make_tutorials()

        # a workaround to reduce the size of the image in the pdf document
        # TODO: v.0.2 probably better solution exists?
        if builder == 'latex':
            self.resize_img(GALLERYDIR, size=580.)

    # ..................................................................................................................
    @staticmethod
    def resize_img(folder, size):
        # image resizing mainly for pdf doc

        for img in iglob(os.path.join(folder, '**', '*.png'), recursive=True):
            if not img.endswith('.png'):
                continue
            filename = os.path.join(folder, img)
            image = imread(filename)
            h, l, c = image.shape
            ratio = 1.
            if l > size:
                ratio = size / l
            if ratio < 1:
                # reduce size
                image_resized = resize(image, (int(image.shape[0] * ratio), int(image.shape[1] * ratio)),
                                       anti_aliasing=True)
                # print(img, 'original:', image.shape, 'ratio:', ratio, " -> ", image_resized.shape)
                imsave(filename, (image_resized * 255.).astype(np.uint8))

    # ..................................................................................................................
    def make_pdf(self):
        # Generate the PDF documentation

        doc_version = self.doc_version
        LATEXDIR = f"{LATEX}/{doc_version}"
        print('Started to build pdf from latex using make.... '
              'Wait until a new message appear (it is a long! compilation) ')

        print('FIRST COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print('MAKEINDEX:')
        sh(f"cd {LATEXDIR}; makeindex spectrochempy.idx")

        print('SECOND COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print("move pdf file in the download dir")
        sh(f"cp {os.path.join(LATEXDIR, PROJECT)}.pdf {DOWNLOADS}/{doc_version}-{PROJECT}.pdf")

    # ..................................................................................................................
    def sync_notebooks(self):
        # Use  jupytext to sync py and ipynb files in userguide and tutorials

        sh.jupytext("--sync", f"{USERGUIDE}", silent=True)
        sh.jupytext("--sync", f"{TUTORIALS}", silent=True)

    # ..................................................................................................................
    def delnb(self):
        # Remove all ipynb before git commit

        for nb in iglob(os.path.join(USERDIR, '**', '*.ipynb'), recursive=True):
            sh.rm(nb)

    # ..................................................................................................................
    def make_tutorials(self):
        # make tutorials.zip

        # clean notebooks output
        for nb in iglob(os.path.join(DOCDIR, '**', '*.ipynb'), recursive=True):
            # This will erase all notebook output
            sh(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}", silent=True)

        # make zip of all ipynb
        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for nb in iglob(os.path.join(path, '**', '*.ipynb'), recursive=True):
                # # remove outputs
                if '.ipynb_checkpoints' in nb:
                    continue
                basename = os.path.basename(nb).split(".ipynb")[0]
                sh(f"jupyter nbconvert {nb} --to notebook"
                   f" --ClearOutputPreprocessor.enabled=True"
                   f" --stdout > out_{basename}.ipynb")
                sh(f"rm {nb}", silent=True)
                sh(f"mv out_{basename}.ipynb {nb}", silent=True)
                arcnb = nb.replace(path, dest)
                ziph.write(nb, arcname=arcnb)

        zipf = zipfile.ZipFile('~notebooks.zip', 'w', zipfile.ZIP_STORED)
        zipdir(USERDIR, 'notebooks', zipf)
        zipdir(os.path.join(GALLERYDIR, 'auto_examples'), os.path.join('notebooks', 'examples'), zipf)
        zipf.close()

        sh(f"mv ~notebooks.zip {DOWNLOADS}/{self.doc_version}-{PROJECT}-notebooks.zip")

    # ..................................................................................................................
    def api_gen(self):
        """
        Generate the API reference rst files
        """

        apigen(SOURCESDIR,
               tocdepth=1,
               force=True,
               includeprivate=False,
               destdir=API,
               exclude_patterns=[
                       'NDArray',
                       'NDComplexArray',
                       'NDIO',
                       'NDPlot',
                       ], )

    # ..................................................................................................................
    def make_redirection_page(self, ):

        html = f"""
        <html>
        <head>
        <title>redirect to the dev version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://{URL_SCPY}/latest">
        </head>
        <body></body>
        </html>
        """
        with open(os.path.join(HTML, 'index.html'), 'w') as f:
            f.write(html)

    # ..................................................................................................................
    def clean(self, builder):
        """
        Clean/remove the built documentation.
        """

        doc_version = self.doc_version

        shutil.rmtree(os.path.join(HTML, doc_version), ignore_errors=True)
        shutil.rmtree(os.path.join(LATEX, doc_version), ignore_errors=True)
        shutil.rmtree(os.path.join(DOCTREES, doc_version), ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)
        shutil.rmtree(GALLERYDIR, ignore_errors=True)

    # ..................................................................................................................
    def make_dirs(self):
        """
        Create the directories required to build the documentation.
        """
        doc_version = self.doc_version

        # Create regular directories.
        build_dirs = [
                os.path.join(DOCTREES, doc_version),
                os.path.join(HTML, doc_version),
                os.path.join(LATEX, doc_version),
                DOWNLOADS,
                os.path.join(DOCDIR, '_static'),
                ]
        for d in build_dirs:
            os.makedirs(d, exist_ok=True)

    # ..................................................................................................................
    def make_changelog(self, milestone="latest"):
        """
        Utility to update changelog (using the GITHUB API)
        """
        if milestone == 'latest':
            # we build the latest
            print("getting latest release tag")
            LATEST = os.path.join(API_GITHUB_URL, "repos", REPO_URI, "releases", "latest")
            tag = json.loads(requests.get(LATEST).text)['tag_name'].split('.')
            milestone = f"{tag[0]}.{tag[1]}.{int(tag[2]) + 1}"  # TODO: this will not work if we change the minor or
            # major

        def get(milestone, label):
            print("getting list of issues with label ", label)
            issues = os.path.join(API_GITHUB_URL, "search", f"issues?q=repo:{REPO_URI}"
                                                            f"+milestone:{milestone}"
                                                            f"+is:closed"
                                                            f"+label:{label}")
            return json.loads(requests.get(issues).text)['items']

        # Create a versionlog file for the current target
        bugs = get(milestone, "bug")
        features = get(milestone, "enhancement")
        tasks = get(milestone, "task")

        with open(os.path.join(TEMPLATES, 'versionlog.rst'), 'r') as f:
            template = Template(f.read())
        out = template.render(target=milestone, bugs=bugs, features=features, tasks=tasks)

        with open(os.path.join(DOCDIR, 'versionlogs', f'versionlog.{milestone}.rst'), 'w') as f:
            f.write(out)

        # make the full version history
        lhist = sorted(iglob(os.path.join(DOCDIR, 'versionlogs', '*.rst')))
        lhist.reverse()
        history = ""
        for filename in lhist:
            if '.'.join(filename.split('.')[-4:-1]) > milestone:
                continue  # do not take into account future version for change log - obviously!
            with open(filename, 'r') as f:
                history += "\n\n"
                nh = f.read().strip()
                vc = ".".join(filename.split('.')[1:4])
                nh = nh.replace(':orphan:', '') #f".. _version_{vc}:")
                history += nh
        history += '\n'

        with open(os.path.join(TEMPLATES, 'changelog.rst'), 'r') as f:
            template = Template(f.read())
        out = template.render(history=history)

        outfile = os.path.join(DOCDIR, 'gettingstarted', 'changelog.rst')
        with open(outfile, 'w') as f:
            f.write(out)

        sh.pandoc(outfile, '-f', 'rst', '-t', 'markdown',  '-o',
                  os.path.join(PROJECTDIR,'CHANGELOG.md'))

        return


Build = Build()

if __name__ == '__main__':
    Build()
