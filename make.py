#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""Clean, build, and release the HTML and PDF documentation for SpectroChemPy.

usage::

    python make.py [options]

where optional parameters indicates which job(s) is(are) to perform.

"""
import argparse
import os
import re
import shutil
import warnings
from glob import iglob
from subprocess import Popen, PIPE
from skimage.transform import resize
from skimage.data import load
from skimage.io import imread, imsave
import numpy as np

from sphinx.application import Sphinx, RemovedInSphinx30Warning, RemovedInSphinx40Warning

from spectrochempy import version

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
TUTORIALS = os.path.join(USERDIR, "tutorials", "*", "*.py")
USERGUIDE = os.path.join(USERDIR, "userguide", "*", "*.py")
API = os.path.join(DOCDIR, 'api', 'generated')
BUILDDIR = os.path.normpath(os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT))
DOCTREES = os.path.normpath(os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT, '~doctrees'))
HTML = os.path.join(BUILDDIR, 'html')
LATEX = os.path.join(BUILDDIR, 'latex')
DOWNLOADS = os.path.join(HTML, 'downloads')

__all__ = []


# ======================================================================================================================
class Build(object):
    
    # ..................................................................................................................
    def __init__(self):
        
        # determine if we are in the developement branch (latest) or master (stable)
        self.doc_version = 'latest' if 'dev' in version else 'stable'
    
    # ..................................................................................................................
    def __call__(self):
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument("-w", "--html", help="create html pages", action="store_true")
        parser.add_argument("-p", "--pdf", help="create pdf pages", action="store_true")
        parser.add_argument("-c", "--clean", help="clean/delete output", action="store_true")
        parser.add_argument("-d", "--deepclean", help="full clean/delete output (reset)", action="store_true")
        parser.add_argument("-s", "--sync", help="sync doc ipynb", action="store_true")
        parser.add_argument("-g", "--git", help="git commit last changes", action="store_true")
        parser.add_argument("-m", "--message", default='DOCS: updated', help='optional commit message')
        parser.add_argument("-a", "--api", help="full regeneration of the api", action="store_true")
        parser.add_argument("-r", "--release", help="release documentation on website", action="store_true")
        args = parser.parse_args()
        
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
            
    @staticmethod
    def _cmd_exec(cmd, shell=None):
        # Private function to execute system command
    
        if shell is not None:
            res = Popen(cmd, shell=shell, stdout=PIPE, stderr=PIPE)
        else:
            res = Popen(cmd, stdout=PIPE, stderr=PIPE)
        output, error = res.communicate()
        if not error:
            v = output.decode("utf-8")
            return v
        else:
            v = error.decode('utf-8')
            if "makeindex" in v:
                return v  # This is not an error! (#TODO: why Popen retrun an error)
            raise RuntimeError(f"{cmd} [FAILED]\n{v}")

    @staticmethod
    def _confirm(action):
        # Ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(f"OK to continue `{action}` [Y/N]? ", ).lower()
        return answer == "y"
    
    # ..................................................................................................................
    def make_docs(self, builder):
        """
        Make the documentation
    
        Parameters
        ----------
        builder: str,
            Type of builder
            
        """
        doc_version = self.doc_version
        
        print(f'building {builder.upper()} documentation ({doc_version.capitalize()} version : {version})')
        
        # recreate dir if needed
        self.clean(builder)
        self.make_dirs()
        srcdir = confdir = DOCDIR
        outdir = f"{BUILDDIR}/{builder}/{doc_version}"
        doctreesdir = f"{BUILDDIR}/~doctrees/{doc_version}"
        
        # regenate api documentation
        if (self.regenerate_api or not os.path.exists(API)):
            self.api_gen()
        
        # run sphinx
        sp = Sphinx(srcdir, confdir, outdir, doctreesdir, builder)
        sp.verbosity = 1
        sp.build()
        res = sp.statuscode
        
        print(f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages are in {os.path.normpath(outdir)}.")
        
        # do some cleaning
        shutil.rmtree(os.path.join('docs', 'auto_examples'), ignore_errors=True)
        
        self.resize_img(os.path.join(DOCDIR,"gallery" ))
        
        if builder == 'html':
            self.update_html_page(outdir)
            self.make_redirection_page()
        
    # ..................................................................................................................
    @staticmethod
    def resize_img(folder):
        for img in iglob(os.path.join(folder, '**', '*.png'), recursive=True):
            if not img.endswith('.png'):
                continue
            filename = os.path.join(folder, img)
            image = imread(filename)
            h, l, c = image.shape
            ratio = 1.
            if float(l)>640.:
                ratio = 640./float(l)
            if ratio < 1:
                # reduce size
                image_resized = resize(image, (int(image.shape[0]*ratio), int(image.shape[1]*ratio)),
                                       anti_aliasing=True)
                print (img, 'original:', image.shape, 'ratio:', ratio, " -> ", image_resized.shape)
                imsave(filename, (image_resized*255.).astype(np.uint8))
                
                
    # ..................................................................................................................
    def make_pdf(self):
        doc_version = self.doc_version
        latexdir = f"{BUILDDIR}/latex/{doc_version}"
        print(
            'Started to build pdf from latex using make.... Wait until a new message appear (it is a long! compilation) ')
        
        output = self._cmd_exec(f'cd {os.path.normpath(latexdir)};'
                           f'lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex',
                           shell=True)
        print('FIRST COMPILATION:', output)
        
        output = self._cmd_exec(f'cd {os.path.normpath(latexdir)};'
                           f'makeindex spectrochempy.idx',
                           shell=True)
        print('MAKEINDEX', output)
        
        output = self._cmd_exec(f'cd {os.path.normpath(latexdir)};'
                           f'lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex',
                           shell=True)
        print('SECOND COMPILTATION:', output)
        
        output = self._cmd_exec(f'cd {os.path.normpath(latexdir)}; '
                           f'cp {PROJECT}.pdf {DOWNLOADS}/scpy.pdf', shell=True)
        print(output)
    
    # ..................................................................................................................
    def sync_notebooks(self):
        # we need to use jupytext to sync py and ipynb files in userguide and tutorials
        cmds = (f"jupytext --sync {USERGUIDE}", f"jupytext --sync {TUTORIALS}")
        for cmd in cmds:
            cmd = cmd.split()
            print(self._cmd_exec(cmd))
    
    # ..................................................................................................................
    def api_gen(self):
        from docs import apigen
        
        # generate API reference
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
        output = self._cmd_exec(cmd)
        print(output)
        
        cmd = "git log -1 --pretty=%B".split()
        output = self._cmd_exec(cmd)
        print('last message: ', output)
        if output.strip() == message:
            v = "--amend"
        else:
            v = "--no-verify"
        
        cmd = f"git commit {v} -m".split()
        cmd.append(message)
        output = self._cmd_exec(cmd)
        print(output)
        
        cmd = "git log -1 --pretty=%B".split()
        output = self._cmd_exec(cmd)
        print('new message: ', output)
        
        # TODO: Automate Tagging?
    
    # ..................................................................................................................
    def make_redirection_page(self, ):
        
        html = """
        <html>
        <head>
        <title>redirect to the stable version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://www.spectrochempy.fr/stable">
        </head>
        <body></body>
        </html>
        """
        with open(os.path.join(HTML, 'index.html'), 'w') as f:
            f.write(html)
    
    # ..................................................................................................................
    def update_html_page(self, outdir):
        """
        Modify page generated with sphinx (TODO: There is porbably a better method using sphinx templates to override
        the themes)
        """
        
        replace = """
                <div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions">
                
                    <span class="rst-current-version" data-toggle="rst-current-version">
                      <span class="fa fa-book">SpectroChemPy</span>
                      v: %s
                      <span class="fa fa-caret-down"></span>
                    </span>
                
                    <div class="rst-other-versions">
                        <dl>
                            <dt>Versions</dt>
                            <dd><a href="/latest/index.html">latest</a></dd>
                            <dd><a href="/stable/index.html">stable</a></dd>
                        </dl>
                
                        <dl>
                            <dt>Downloads</dt>
                            <dd><a href="/downloads/scpy.pdf">pdf</a></dd>
                            <!--<dd><a href="/downloads/scpy_doc.zip">htmlzip</a></dd>-->
                            <!--<dd><a href="/tutorials/">tutorials</a></dd>-->
                        </dl>
                
                        <dl>
                            <dt>Sources on bitBucket</dt>
                            <dd><a href="https://bitbucket.org/spectrocat/spectrochempy/src/master/">master</a></dd>
                            <dd><a href="https://bitbucket.org/spectrocat/spectrochempy/src/develop/">develop</a></dd>
                        </dl>
                
                        <hr/>
                        
                    </div>
                </div>
            
                <script type="text/javascript" id="already-corrected">
                    jQuery(function () {
                        SphinxRtdTheme.Navigation.enable(true);
                    });
                </script>
                """
        # modify all html files
        for filename in iglob(os.path.join(outdir, '**', '*.html'), recursive=True):
            with open(filename, "r") as f:
                txt = f.read()
            doc_version = self.doc_version
            regex = r"(<script type=\"text\/javascript\">.*SphinxRtdTheme.*script>)"
            result = re.sub(regex, replace % doc_version, txt, 0, re.MULTILINE | re.DOTALL)
            with open(filename, "w") as f:
                f.write(result)
    
    # ..................................................................................................................
    def release(self):
        """
        Release/publish the documentation to the webpage.
        """
        doc_version = self.doc_version
        
        # upload docs to the remote web server
        if SERVER:
            
            print("uploads to the server of the html/pdf files")
            
            FROM = os.path.join(HTML, '*')
            TO = os.path.join(PROJECT, 'html')
            cmd = f'rsync -e ssh -avz  --exclude="~*" {FROM} {SERVER}:{TO}'
            output = self._cmd_exec(cmd, shell=True)
            print(output)
        
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
    
    def deepclean(self):
        
        doc_version = self.doc_version
        
        shutil.rmtree(os.path.join(DOCTREES, doc_version), ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)
    
    # ..................................................................................................................
    def make_dirs(self):
        """
        Create the directories required to build the documentation.
        """
        doc_version = self.doc_version
        
        # Create regular directories.
        build_dirs = [os.path.join(DOCTREES, doc_version),
                      os.path.join(HTML, doc_version),
                      os.path.join(LATEX, doc_version),
                      DOWNLOADS,
                      os.path.join(DOCDIR, '_static'),
                      ]
        for d in build_dirs:
            os.makedirs(d, exist_ok=True)


Build = Build()

if __name__ == '__main__':
    Build()
