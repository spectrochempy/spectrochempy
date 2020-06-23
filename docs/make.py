#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Clean, build, and release the HTML and PDF documentation for SpectroChemPy.

usage::

    python make.py [options]

where optional parameters indicates which job(s) is(are) to perform.

"""

import argparse
import os
import shutil
import sys
import warnings
import zipfile
from glob import iglob

import json5 as json
import numpy as np
import requests
from jinja2 import Template
from skimage.io import imread, imsave
from skimage.transform import resize
from spectrochempy import version
from spectrochempy.utils import sh
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx40Warning

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx50Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)


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
class BuildDocumentation(object):

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
        for nbch in iglob(os.path.join(USERDIR, '**', '.ipynb_checkpoints'), recursive=True):
            sh(f'rm -r {nbch}')

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
    def make_redirection_page(self, ):
        # create an index page a the site root to redirect to latest version

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
        # Clean/remove the built documentation.

        doc_version = self.doc_version

        shutil.rmtree(os.path.join(HTML, doc_version), ignore_errors=True)
        shutil.rmtree(os.path.join(LATEX, doc_version), ignore_errors=True)
        shutil.rmtree(os.path.join(DOCTREES, doc_version), ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)
        shutil.rmtree(GALLERYDIR, ignore_errors=True)

    # ..................................................................................................................
    def make_dirs(self):
        # Create the directories required to build the documentation.

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
        # Utility to update changelog (using the GITHUB API)

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
                nh = nh.replace(':orphan:', '')
                history += nh
        history += '\n'

        with open(os.path.join(TEMPLATES, 'changelog.rst'), 'r') as f:
            template = Template(f.read())
        out = template.render(history=history)

        outfile = os.path.join(DOCDIR, 'api', 'changelog.rst')
        with open(outfile, 'w') as f:
            f.write(out)

        sh.pandoc(outfile, '-f', 'rst', '-t', 'markdown', '-o',
                  os.path.join(PROJECTDIR, 'CHANGELOG.md'))

        return


Build = BuildDocumentation()

if __name__ == '__main__':
    Build()
