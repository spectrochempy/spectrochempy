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

    python build_docs.py [clean notebooks html epub pdf release]

where optional parameters idincates which job to perfom.

"""
import shutil
from subprocess import Popen, PIPE
import re
import os

import argparse

from sphinx.application import Sphinx, RemovedInSphinx30Warning, RemovedInSphinx40Warning

import warnings
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

class Build(object):
    
    # ..................................................................................................................
    def __init__(self):
        self.regenerate_api = False
    
    # ..................................................................................................................
    def __call__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-w", "--html", help="create html pages", action="store_true")
        parser.add_argument("-p", "--pdf", help="create pdf pages", action="store_true")
        parser.add_argument("-e", "--epub", help="create epub pages", action="store_true")
        parser.add_argument("-d", "--delete", help="clean/delete output", action="store_true")
        parser.add_argument("-s", "--sync", help="sync doc ipynb", action="store_true")
        parser.add_argument("-c", "--commit", help="commit last changes", action="store_true")
        parser.add_argument("-m", "--message", default='DOCS: updated', help='optional commit message')
        args = parser.parse_args()

        if args.sync:
            self.sync_notebooks()
        if args.commit:
            self.gitcommit(args.message)
            
        
    # ..................................................................................................................
    def sync_notebooks(self):
        # we need to use jupytext to sync py and ipynb files in userguide and tutorials
        cmds = (f"jupytext --sync {USERGUIDE}", f"jupytext --sync {TUTORIALS}")
        for cmd in cmds:
            res = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            output, error = res.communicate()
            if not error:
                print(output.decode("utf-8"))
            else:
                print(error.decode("utf-8"))
    
    # ..................................................................................................................
    def api_gen(self, force=False):
        from docs import apigen
        # generate API reference
        apigen.main(SOURCESDIR,
                    tocdepth=1,
                    force=force,
                    includeprivate=True,
                    destdir=API,
                    exclude_patterns=[
                        'NDArray',
                        'NDComplexArray',
                        'NDIO',
                        'NDPlot',
                    ],)
        
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
            
            pipe = Popen(["git", "add", "-A"], stdout=PIPE, stderr=PIPE)
            (so, serr) = pipe.communicate()
            output = so.decode("ascii")
            print(output)
            
            pipe = Popen(["git", "log", "-1", "--pretty=%B"], stdout=PIPE, stderr=PIPE)
            (so, serr) = pipe.communicate()
            output = so.decode("ascii")
            print('last message', output)
        
            pipe = Popen(["git", "commit", "--no-verify", f"-m {message}"],
                         stdout=PIPE, stderr=PIPE)
            (so, serr) = pipe.communicate()
            output = so.decode("ascii")
            print(output)
            print('\nCommit last changes done')

            #Automatically Tag?
        
    # ..................................................................................................................
    def make_redirection_page(self):

        html = """
        <html>
        <head>
        <title>redirect to the stable version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=https://www.spectrochempy.fr/stable">
        </head>
        <body></body>
        </html>
        """

    # ..................................................................................................................
    def update_html_page(outdir, version):
        """
        Modify page generated with sphinx (TODO: There is porbably a better method using sphinx templates to override
        the themes)
        """
        
        replace="""
                <div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions">
                
                    <span class="rst-current-version" data-toggle="rst-current-version">
                      <span class="fa fa-book">SpectroChemPy</span>
                      v: %s
                      <span class="fa fa-caret-down"></span>
                    </span>
                
                    <div class="rst-other-versions">
                        <dl>
                            <dt>Versions</dt>
                            <dd><a href="https://spectrochempy.fr/html/latest/">latest</a></dd>
                            <dd><a href="https://spectrochempy.fr/html/stable/">stable</a></dd>
                        </dl>
                
                        <dl>
                            <dt>Downloads</dt>
                            <dd><a href="https://spectrochempy.fr/pdf/stable/">pdf</a></dd>
                            <dd><a href="https://spectrochempy.fr/htmlzip/stable/">htmlzip</a></dd>
                            <dd><a href="https://spectrochempy.fr/epub/stable/">epub</a></dd>
                            <dd><a href="https://spectrochempy.fr/tutorials/">tutorials</a></dd>
                        </dl>
                
                        <dl>
                            <dt>Sources on bitBucket</dt>
                            <dd><a href="https://bitbucket.org/spectrocat/spectrochempy/src/master/">master</a></dd>
                            <dd><a href="https://bitbucket.org/spectrocat/spectrochempy/src/develop/">develop</a></dd>
                        </dl>
                
                        <hr/>
                        
                    </div>
                </div>
            
                <script type="text/javascript">
                    jQuery(function () {
                        SphinxRtdTheme.Navigation.enable(true);
                    });
                </script>
                """
        
        with open(os.path.join(outdir, 'index.html'), "r") as f:
            txt = f.read()
        
        regex = r"(<script type=\"text\/javascript\">.*SphinxRtdTheme.*script>)"
        result = re.sub(regex, replace % version, txt, 0, re.MULTILINE | re.DOTALL)
        
        if result:
            
            with open(os.path.join(outdir, 'index.html'), "w") as f:
                f.write(result)
    
    # ..................................................................................................................
    def do_release(self, version):
        """
        Release/publish the documentation to the webpage.
        """
        
        # make the doc
        # make_docs(*args)
        
        # upload docs to the remote web server
        if SERVER:
            
            print("uploads to the server of the html/pdf/epub files")
            
            for item in ['html','pdf','epub']:
                FROM = os.path.join(BUILDDIR, item, version, '*')
                TO = os.path.join(PROJECT, item, version)
                cmd = f'rsync -e ssh -avz  --exclude="~*" {FROM} {SERVER}:{TO}'.split()
                pipe = Popen(*cmd, stdout=PIPE, stderr=PIPE)
                (so, serr) = pipe.communicate()
                output = so.decode("ascii")
                print(output)
            
        else:
            
            print('Cannot find the upload server : {}!'.format(SERVER))
    
    # ..................................................................................................................
    def clean(self, version):
        """
        Clean/remove the built documentation.
        """
        shutil.rmtree(os.path.join(BUILDDIR,'html', version), ignore_errors=True)
        shutil.rmtree(os.path.join(BUILDDIR,'pdf', version), ignore_errors=True)
        shutil.rmtree(os.path.join(BUILDDIR,'latex', version), ignore_errors=True)
        shutil.rmtree(os.path.join(BUILDDIR,'epub', version), ignore_errors=True)
        shutil.rmtree(os.path.join(DOCTREES, version), ignore_errors=True)
        shutil.rmtree(API, ignore_errors=True)
    
    # ..................................................................................................................
    def make_dirs(self, version):
        """
        Create the directories required to build the documentation.
        """
        
        # Create regular directories.
        build_dirs = [os.path.join(BUILDDIR, '~doctrees', version),
                      os.path.join(BUILDDIR, 'html', version),
                      os.path.join(BUILDDIR, 'latex', version),
                      os.path.join(BUILDDIR, 'pdf', version),
                      os.path.join(BUILDDIR, 'epub', version),
                      os.path.join(DOCDIR, '_static'),
                      ]
        for d in build_dirs:
            os.makedirs(d, exist_ok=True)


Build = Build()

if __name__ == '__main__':
    
    Build()
