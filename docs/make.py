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
import shutil
import sys
import warnings
import zipfile
import json5 as json
import numpy as np
import requests
from jinja2 import Template
from pathlib import Path

from skimage.io import imread, imsave
from skimage.transform import resize
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx50Warning, RemovedInSphinx40Warning

from spectrochempy import version
from spectrochempy.utils import sh

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx50Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)

# CONSTANT
PROJECT = "spectrochempy"
REPO_URI = f"spectrochempy/{PROJECT}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "spectrochempy.github.io/spectrochempy"

# PATHS
HOME = Path().home()
DOCDIR = Path(__file__).parent
PROJECTDIR = DOCDIR.parent
SOURCESDIR = PROJECTDIR / PROJECT
USERDIR = DOCDIR / 'user'
TEMPLATES = DOCDIR / '_templates'
USERGUIDE = list(map(str, (USERDIR / "userguide").glob("**/*.py")))
API = DOCDIR / 'reference' / 'generated'
GALLERYDIR = DOCDIR / "gallery"
DOCREPO =  HOME / "spectrochempy_docs"
DOCTREES = DOCREPO / "~doctrees"
HTML =  DOCREPO / "html"
LATEX = DOCREPO / 'latex'
DOWNLOADS = HTML / 'downloads'

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
                            action="store_true")
        parser.add_argument("--all", help="Build all docs", action="store_true")

        args = parser.parse_args()

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            return

        self.regenerate_api = args.api

        if args.sync:
            self.sync_notebooks()
        if args.changelogs:
            self.make_changelog()
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

        BUILDDIR = DOCREPO / builder
        print(f'\n{"*"*80}\n'
              f'building {builder.upper()} documentation ({doc_version.capitalize()} version : {version})'
              f'\n{"*"*80}\n')

        self.make_changelog()

        # recreate dir if needed
        if clean:
            self.clean(builder)
        self.make_dirs()

        self.sync_notebooks()

        # run sphinx
        srcdir = confdir = DOCDIR
        outdir = f"{BUILDDIR}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"
        sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
        sp.verbosity = 1
        sp.build()

        print(f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
              f"are in {outdir}.")

        # do some cleaning
        if clean:
            shutil.rmtree( GALLERYDIR / 'auto_examples', ignore_errors=True)

        if builder == 'html':
            self.make_redirection_page()
            self.make_tutorials()

        # a workaround to reduce the size of the image in the pdf document
        # TODO: v.0.2 probably better solution exists?
        if builder == 'latex':
            self.resize_img(GALLERYDIR, size=580.)

        # when it is terminated suppress all ipynb to avoid problems with
        # jupyter lab abd jupytex
        self.delnb()

    # ..................................................................................................................
    @staticmethod
    def resize_img(folder, size):
        # image resizing mainly for pdf doc

        for img in folder.rglob('**/*.png'):
            if not img.endswith('.png'):
                continue
            filename = folder / img
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
        LATEXDIR = LATEX / doc_version
        print('Started to build pdf from latex using make.... '
              'Wait until a new message appear (it is a long! compilation) ')

        print('FIRST COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print('MAKEINDEX:')
        sh(f"cd {LATEXDIR}; makeindex spectrochempy.idx")

        print('SECOND COMPILATION:')
        sh(f"cd {LATEXDIR}; lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex")

        print("move pdf file in the download dir")
        sh(f"cp {LATEXDIR / PROJECT}.pdf {DOWNLOADS}/{doc_version}-{PROJECT}.pdf")

    # ..................................................................................................................
    def sync_notebooks(self):
        # Use  jupytext to sync py and ipynb files in userguide and tutorials

        for item in USERGUIDE:
            sh.jupytext("--sync" ,item ,silent=False)

    # ..................................................................................................................
    def delnb(self):
        # Remove all ipynb before git commit

        for nb in USERDIR.rglob('**/*.ipynb'):
            sh.rm(nb)
        for nbch in USERDIR.glob('**/.ipynb_checkpoints'):
            sh(f'rm -r {nbch}')

    # ..................................................................................................................
    def make_tutorials(self):
        # make tutorials.zip

        # clean notebooks output
        for nb in DOCDIR.rglob('**/*.ipynb'):
            # This will erase all notebook output
            sh(f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}", silent=True)

        # make zip of all ipynb
        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for nb in path.rglob('**/*.ipynb'):
                if '.ipynb_checkpoints' in nb.parent.suffix:
                    continue
                basename = nb.stem
                sh(f"jupyter nbconvert {nb} --to notebook"
                   f" --ClearOutputPreprocessor.enabled=True"
                   f" --stdout > out_{basename}.ipynb")
                sh(f"rm {nb}", silent=True)
                sh(f"mv out_{basename}.ipynb {nb}", silent=True)
                arcnb = str(nb).replace(str(path), str(dest))
                ziph.write(nb, arcname=arcnb)

        zipf = zipfile.ZipFile('~notebooks.zip', 'w', zipfile.ZIP_STORED)
        zipdir(USERDIR, 'notebooks', zipf)
        zipdir(GALLERYDIR / 'auto_examples', Path('notebooks') / 'examples', zipf)
        zipf.close()

        sh(f"mv ~notebooks.zip {DOWNLOADS}/{self.doc_version}-{PROJECT}-notebooks.zip")

    # ..................................................................................................................
    def make_redirection_page(self, ):
        # create an index page a the site root to redirect to stable version

        html = f"""
        <html>
        <head>
        <title>redirect to the dev version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://{URL_SCPY}/stable">
        </head>
        <body></body>
        </html>
        """
        with open(HTML / 'index.html', 'w') as f:
            f.write(html)

    # ..................................................................................................................
    def clean(self, builder):
        # Clean/remove the built documentation.

        doc_version = self.doc_version

        shutil.rmtree(HTML / doc_version, ignore_errors=True)
        shutil.rmtree(LATEX / doc_version, ignore_errors=True)
        shutil.rmtree(DOCTREES / doc_version, ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)
        shutil.rmtree(GALLERYDIR, ignore_errors=True)

    # ..................................................................................................................
    def make_dirs(self):
        # Create the directories required to build the documentation.

        doc_version = self.doc_version

        # Create regular directories.
        build_dirs = [
                DOCTREES / doc_version,
                HTML / doc_version,
                LATEX / doc_version,
                DOWNLOADS,
                DOCDIR / '_static',
                ]
        for d in build_dirs:
            Path.mkdir(d, exist_ok=True)

    # ..................................................................................................................
    def make_changelog(self):
        # Utility to update changelog (using the GITHUB API)

        print(f'\n{"-"*80}\nMake Change  logs\n')
        print("getting latest release tag")
        LATEST = f'{API_GITHUB_URL}/repos/{REPO_URI}/releases/latest'
        tag = json.loads(requests.get(LATEST).text)['tag_name']

        milestone = self.doc_version

        if milestone == 'latest':
            # we build the latest
            tag = tag.split('.')
            milestone = f"{tag[0]}.{tag[1]}.{int(tag[2]) }"  # TODO: this will not work if we change the minor or
            # major
        else:
            milestone = tag

        def get(milestone, label):
            print("getting list of issues with label ", label)
            issues = API_GITHUB_URL + "/search/issues?q=repo:" + REPO_URI
            issues += "+milestone:" + milestone + "+is:closed"
            if label != "pr":
                issues += "+label:" + label
            else:
                issues += "+type:pr"
            return json.loads(requests.get(issues).text)['items']

        # Create a versionlog file for the current target
        prs = get(milestone, "pr")
        bugs = get(milestone, "bug")
        features = get(milestone, "enhancement")
        tasks = get(milestone, "task")

        with open(TEMPLATES / 'versionlog.rst', 'r') as f:
            template = Template(f.read())
        out = template.render(target=milestone, prs=prs, bugs=bugs, features=features, tasks=tasks)

        change = DOCDIR / 'versionlogs' / f'versionlog.{milestone}.rst'
        with open(change, 'w') as f:
            f.write(out)
            print(f'\n{"-" * 80}\nversion {milestone} log written to:\n{change}\n{"-"*80}')

        # make the full version history
        lhist = sorted(DOCDIR.glob('versionlogs/*.rst'))
        lhist.reverse()
        history = ""
        for filename in lhist:
            if filename.stem.replace('versionlog.','') > milestone:
                continue  # do not take into account future version for change log - obviously!
            with open(filename, 'r') as f:
                history += "\n\n"
                nh = f.read().strip()
                # vc = ".".join(filename.split('.')[1:4])
                nh = nh.replace(':orphan:', '')
                history += nh
        history += '\n'

        with open( TEMPLATES / 'changelog.rst', 'r') as f:
            template = Template(f.read())
        out = template.render(history=history)

        outfile = DOCDIR / 'gettingstarted' / 'changelog.rst'
        with open(outfile, 'w') as f:
            f.write(out)
            print(f'`Complete what\'s new` log written to:\n{outfile}\n{"-" * 80}')

        sh.pandoc(outfile, '-f', 'rst', '-t', 'markdown', '-o', PROJECTDIR / 'CHANGELOG.md')

        return


Build = BuildDocumentation()

if __name__ == '__main__':
    from os import environ
    environ['DOC_BUILDING'] = 'yes'
    Build()
    del environ['DOC_BUILDING']
