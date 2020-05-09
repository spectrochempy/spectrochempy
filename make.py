#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
Clean, build, and release the HTML and PDF documentation for SpectroChemPy.

usage::

    python make.py [options]

where optional parameters indicates which job(s) is(are) to perform.

"""

import argparse
import os, sys
import re
import shutil
import warnings
import zipfile
from glob import iglob
from subprocess import Popen, PIPE
from skimage.transform import resize
from skimage.io import imread, imsave
import numpy as np
from jinja2 import Template
import pandas as pd

from sphinx.application import Sphinx, RemovedInSphinx30Warning, RemovedInSphinx40Warning

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx30Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)
warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)

SERVER = os.environ.get('SERVER_FOR_LCS', None)
PROJECT = "spectrochempy"
PROJECTDIR = os.path.dirname(os.path.abspath(__file__))
SOURCESDIR = os.path.join(PROJECTDIR, "spectrochempy")
DOCDIR = os.path.join(PROJECTDIR, "docs")
USERDIR = os.path.join(PROJECTDIR, "docs", "user")
TEMPLATES = os.path.join(DOCDIR, '_templates')
TUTORIALS = os.path.join(USERDIR, "tutorials", "*", "*.py")
USERGUIDE = os.path.join(USERDIR, "userguide", "*", "*.py")
API = os.path.join(DOCDIR, 'api', 'generated')
BUILDDIR = os.path.normpath(os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT))
DOCTREES = os.path.normpath(os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT, '~doctrees'))
HTML = os.path.join(BUILDDIR, 'html')
LATEX = os.path.join(BUILDDIR, 'latex')
DOWNLOADS = os.path.join(HTML, 'downloads')
GALLERYDIR = os.path.join(DOCDIR,"gallery")

__all__ = []


# ======================================================================================================================
class Build(object):
    
    # ..................................................................................................................
    def __init__(self):
        
        self._doc_version = None
    
    # ..................................................................................................................
    @property
    def doc_version(self):
        from spectrochempy import version, release
    
        if self._doc_version is None:
            # determine if we are in the developement branch (dev) or master (stable)
            self._doc_version = 'dev' if 'dev' in version else 'stable'
        
        return self._doc_version
    
    # ..................................................................................................................
    def __call__(self):
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument("-H", "--html", help="create html pages", action="store_true")
        parser.add_argument("-P", "--pdf", help="create pdf manual", action="store_true")
        parser.add_argument("--tutorials", help="zip notebook tutorials for downloads", action="store_true")
        parser.add_argument("--clean", help="clean/delete html or latex output", action="store_true")
        parser.add_argument("--deepclean", help="full clean/delete output (reset fro a full regenration of the documentation)", action="store_true")
        parser.add_argument("--sync", help="sync doc ipynb using jupytext", action="store_true")
        parser.add_argument("--git", help="git commit last changes", action="store_true")
        parser.add_argument("-m", "--message", default='DOCS: updated', help='optional git commit message')
        parser.add_argument("--api", help="execute a full regeneration of the api", action="store_true")
        parser.add_argument("-R", "--release", help="release the current version documentation on website", action="store_true")
        parser.add_argument("--changelogs", help="update changelogs using the redmine issues status", action="store_true")
        parser.add_argument("--conda", help="make a conda package", action="store_true")
        parser.add_argument("--upload", help="upload conda and pypi package to the corresponding repositories", action="store_true")
        
        args = parser.parse_args()
        
        if len(sys.argv)==1:
            parser.print_help(sys.stderr)
            return
        
        self.regenerate_api = args.api
        
        if args.sync:
            self.sync_notebooks()
        if args.git:
            if self._confirm('COMMIT lAST CHANGES'):
                self.gitcommit(args.message)
        if args.clean and args.html:
            self.clean('html')
        if args.clean and args.pdf:
            self.clean('latex')
        if args.deepclean:
            if self._confirm('DEEP CLEAN'):
                self.deepclean()
        if args.html:
            self.make_docs('html')
        if args.pdf:
            self.make_docs('latex')
            self.make_pdf()
        if args.release:
            self.release()
        if args.tutorials:
            self.make_tutorials()
        if args.changelogs:
            self.make_changelog()
        if args.conda:
            self.make_conda_and_pypi()
        if args.upload:
            self.upload()
            
    @staticmethod
    def _cmd_exec(cmd, shell=None):
        # Private function to execute system command
        print(cmd)
        if shell is not None:
            res = Popen(cmd, shell=shell, stdout=PIPE, stderr=PIPE)
        else:
            res = Popen(cmd, stdout=PIPE, stderr=PIPE)
        output, error = res.communicate()
        if not error:
            v = output.decode("utf-8")
            print(v)
        else:
            v = error.decode('utf-8')
            if "makeindex" not in v and "NbConvertApp" not in v:
                raise RuntimeError(f"{cmd} [FAILED]\n{v}")
            print(v)
        return v

    @staticmethod
    def _confirm(action):
        # private method to ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(f"OK to continue `{action}` Y[es]/[N[o] ? ", ).lower()
        return answer[:1] == "y"
    
    # ..................................................................................................................
    def make_docs(self, builder='html', clean=False):
        """
        Make the html or latex documentation
    
        Parameters
        ----------
        builder: str, optional, default='html'
            Type of builder
            
        """
        from spectrochempy import version, release

        doc_version = self.doc_version
        
        if builder not in ['html', 'latex']:
            raise ValueError('Not a supported builder: Must be "html" or "latex"')
            
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
        outdir = f"{BUILDDIR}/{builder}/{doc_version}"
        doctreesdir = f"{BUILDDIR}/~doctrees/{doc_version}"
        sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
        sp.verbosity = 1
        sp.build()
        res = sp.statuscode
        
        print(f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
              f"are in {os.path.normpath(outdir)}.")
        
        # do some cleaning
        shutil.rmtree(os.path.join('docs', 'auto_examples'), ignore_errors=True)
        
        if builder == 'html':
            self.make_redirection_page()
            self.make_tutorials()
            
        # a workaround to reduce the size of the image in the pdf document
        # TODO: v.0.2 probably better solution exists?
        if builder =='latex':
            self.resize_img(GALLERYDIR, size=580.)
        
        
        
    # ..................................................................................................................
    @staticmethod
    def resize_img(folder, size):
        for img in iglob(os.path.join(folder, '**', '*.png'), recursive=True):
            if not img.endswith('.png'):
                continue
            filename = os.path.join(folder, img)
            image = imread(filename)
            h, l, c = image.shape
            ratio = 1.
            if l>size:
                ratio = size/l
            if ratio < 1:
                # reduce size
                image_resized = resize(image, (int(image.shape[0]*ratio), int(image.shape[1]*ratio)),
                                       anti_aliasing=True)
                print (img, 'original:', image.shape, 'ratio:', ratio, " -> ", image_resized.shape)
                imsave(filename, (image_resized*255.).astype(np.uint8))
                
                
    # ..................................................................................................................
    def make_pdf(self):
        """
        Generate the PDF documentation
        
        """
        doc_version = self.doc_version
        latexdir = f"{BUILDDIR}/latex/{doc_version}"
        print('Started to build pdf from latex using make.... '
              'Wait until a new message appear (it is a long! compilation) ')

        print('FIRST COMPILATION:')
        CMD = f'cd {os.path.normpath(latexdir)};lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex'
        self._cmd_exec(CMD,shell=True)
        
        print('MAKEINDEX:')
        CMD = f'cd {os.path.normpath(latexdir)}; makeindex spectrochempy.idx'
        self._cmd_exec(CMD, shell=True)

        print('SECOND COMPILATION:')
        CMD = f'cd {os.path.normpath(latexdir)};lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex'
        self._cmd_exec(CMD, shell=True)
        
        CMD = f'cd {os.path.normpath(latexdir)}; cp {PROJECT}.pdf {DOWNLOADS}/{doc_version}-{PROJECT}.pdf'
        self._cmd_exec(CMD, shell=True)
    
    # ..................................................................................................................
    def sync_notebooks(self):
        """
        Use  jupytext to sync py and ipynb files in userguide and tutorials
        
        """
        cmds = (f"jupytext --sync {USERGUIDE}", f"jupytext --sync {TUTORIALS}")
        for cmd in cmds:
            cmd = cmd.split()
            self._cmd_exec(cmd)
    
    # ..................................................................................................................
    def make_tutorials(self):
        """
        
        Returns
        -------

        """

        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for nb in iglob(os.path.join(path, '**', '*.ipynb'), recursive=True):
                arcnb = nb.replace(path, dest)
                ziph.write(nb, arcname=arcnb)

        zipf = zipfile.ZipFile('~notebooks.zip', 'w', zipfile.ZIP_STORED)
        zipdir(USERDIR, 'notebooks', zipf)
        zipdir(os.path.join(GALLERYDIR, 'auto_examples'), os.path.join('notebooks', 'examples'), zipf)
        zipf.close()

        CMD = f'mv ~notebooks.zip {DOWNLOADS}/{self.doc_version}-{PROJECT}-notebooks.zip'
        self._cmd_exec(CMD, shell=True)


    # ..................................................................................................................
    def api_gen(self):
        """
        Generate the API reference rst files
        """
        from docs import apigen
        
        apigen.main(SOURCESDIR,
                    tocdepth=1,
                    force=self.regenerate_api,
                    includeprivate=True,
                    destdir=API,
                    exclude_patterns=[
                        'NDArray',
                        'NDComplexArray',
                        'NDIO',
                        'NDPlot',
                    ], )
    
    # ..................................................................................................................
    def gitstatus(self):
        pipe = Popen(["git", "status"], stdout=PIPE, stderr=PIPE)
        (so, serr) = pipe.communicate()
        if "nothing to commit" in so.decode("ascii"):
            return True
        return False
    
    # ..................................................................................................................
    def gitcommit(self, message):
        clean = self.gitstatus()
        if clean:
            return
        
        cmd = "git add -A".split()
        self._cmd_exec(cmd)
        
        cmd = "git log -1 --pretty=%B".split()
        output = self._cmd_exec(cmd)
        if output.strip() == message:
            v = "--amend"
        else:
            v = "--no-verify"
        
        cmd = f"git commit {v} -m".split()
        cmd.append(message)
        self._cmd_exec(cmd)
        
        cmd = "git log -1 --pretty=%B".split()
        self._cmd_exec(cmd)
        
        # TODO: Automate Tagging?
    
    # ..................................................................................................................
    def make_redirection_page(self, ):
        
        html = """
        <html>
        <head>
        <title>redirect to the dev version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://www.spectrochempy.fr/dev">
        </head>
        <body></body>
        </html>
        """
        with open(os.path.join(HTML, 'index.html'), 'w') as f:
            f.write(html)
    
    # ..................................................................................................................
    def release(self):
        """
        Release/publish the documentation to the webpage.
        """
    
        # upload docs to the remote web server
        if SERVER:
            
            print("uploads to the server of the html/pdf files")
            
            FROM = os.path.join(HTML, '*')
            TO = os.path.join(PROJECT, 'html')
            cmd = f'rsync -e ssh -avz  --exclude="~*" {FROM} {SERVER}:{TO}'
            self._cmd_exec(cmd, shell=True)
            
        
        else:
            
            print('Cannot find the upload server : {}!'.format(SERVER))
    
    # ..................................................................................................................
    def clean(self, builder):
        """
        Clean/remove the built documentation.
        """
        
        doc_version = self.doc_version
        
        if builder == 'html':
            shutil.rmtree(os.path.join(HTML, doc_version), ignore_errors=True)
        if builder == 'latex':
            shutil.rmtree(os.path.join(LATEX, doc_version), ignore_errors=True)
    
    # ..................................................................................................................
    def deepclean(self):
        
        doc_version = self.doc_version
        
        shutil.rmtree(os.path.join(DOCTREES, doc_version), ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)

        # clean notebooks output
        for nb in iglob(os.path.join(DOCDIR, '**', '*.ipynb'), recursive=True):
            # This will erase all notebook output
            self._cmd_exec(
                f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {nb}',
                shell=True)
            
    
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
    def make_changelog(self):
        """
        Utility to update changelog
        """
        csv = "https://redmine.spectrochempy.fr/projects/spectrochempy/issues.csv?" \
              "c%5B%5D=tracker" \
              "&c%5B%5D=status" \
              "&c%5B%5D=category" \
              "&c%5B%5D=priority" \
              "&c%5B%5D=subject" \
              "&c%5B%5D=fixed_version" \
              "&f%5B%5D=status_id" \
              "&f%5B%5D=" \
              "&group_by=" \
              "&op%5Bstatus_id%5D=%2A" \
              "&set_filter=1" \
              "&sort=id%3Adesc"

        from spectrochempy import version, release

        issues = pd.read_csv(csv, encoding = "ISO-8859-1")
        doc_version = self.doc_version
        target = version.split('-')[0] if doc_version == 'dev' else release
        changes = issues[issues['Target version']==target]
        
        # Create a versionlog file for the current target
        bugs = changes[changes['Tracker']=='Bug']
        features = changes[changes['Tracker']=='Feature']
        tasks = changes[changes['Tracker']=='Task']
        
        with open(os.path.join(TEMPLATES, 'versionlog.rst'), 'r') as f:
            template = Template(f.read())
        out = template.render(target=target, bugs=bugs, features=features, tasks=tasks)
        
        with open(os.path.join(DOCDIR, 'versionlogs', f'versionlog.{target}.rst'), 'w') as f:
            f.write(out)
            
        # make the full version history
        
        lhist = sorted(iglob(os.path.join(DOCDIR, 'versionlogs', '*.rst')))
        lhist.reverse()
        history = ""
        for filename in lhist:
            with open(filename, 'r') as f:
                history +="\n\n"
                nh = f.read().strip()
                vc = ".".join(filename.split('.')[1:4])
                nh = nh.replace(':orphan:',f".. _version_{vc}:")
                history += nh
        history += '\n'

        with open(os.path.join(TEMPLATES, 'changelog.rst'), 'r') as f:
            template = Template(f.read())
        out = template.render(history=history)

        with open(os.path.join(DOCDIR, 'gettingstarted', 'changelog.rst'), 'w') as f:
            f.write(out)
            
        return
    
    # ..................................................................................................................
    def uploadpypi(self):
        print('UPLOADING TO PYPI')
        CMD ='twine upload --repository pypi dist/*'
        self._cmd_exec(CMD, shell=True)
        
    # ..................................................................................................................
    def make_conda_and_pypi(self):
        """
        
        Parameters
        ----------
        tag

        Returns
        -------

        """
        
        CMDS =  ['print:CONDA PACKAGE UPDATING.... (please wait!)']
        
        CMDS += ["conda config --add channels cantera"]
        CMDS += ["conda config --add channels spectrocat"]
        CMDS += ["conda config --add channels conda-forge"]
        CMDS += ["conda config --set channel_priority strict"]
        CMDS += ['conda config --set anaconda_upload no']
        
        #CMDS += ['conda init zsh']    # normally to adapt depending on shell present in your OS
        #CMDS += ["conda deactivate"]  # # Prefer to work in a clean base environment to better detect incompatibilities
        CMDS += ['conda update conda']
        CMDS += ['conda install pip setuptools wheel twine conda-build conda-verify anaconda-client-y']
        CMDS += ['conda update pip setuptools wheel twine conda-build conda-verify anaconda-client']
        
        CMDS += ['print:CREATING PYPI DISTRIBUTION PACKAGE....']
        CMDS += ['python setup.py sdist bdist_wheel']
        CMDS += ['twine check dist/*']
        
        CMDS += ['print:BUILDING THE CONDA PACKAGE...']
        CMDS += ['conda build recipe/spectrochempy']
        

        #CMDS += ['print:The Conda package is here -> ']
        #CMDS += ['conda build recipe/spectrochempy --output']

        CMDS += ['conda build purge']
        for CMD in CMDS:
            if CMD.startswith('print:'):
                print(CMD[6:])
            else:
                try:
                    self._cmd_exec(CMD, shell=True)
                except RuntimeError as e:
                    if "Your shell has not been properly configured to use 'conda activate'" in e.args[0]:
                        raise e
                    else:
                        print(e.args[0])


        # anaconda upload --user spectrocat ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2 --force
        #
        # conda convert --platform linux-64 ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2
        # anaconda upload --user spectrocat linux-64/spectrochempy-$1.tar.bz2 --force
        #
        # conda convert --platform win-64 ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2
        # anaconda upload --user spectrocat win-64/spectrochempy-$1.tar.bz2 --force



Build = Build()

if __name__ == '__main__':
    Build()
