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

    python builddocs.py [clean html pdf release]

where optional parameters idincates which job to perfom.

"""
import shutil
import subprocess
import re

from sphinx.application import Sphinx, RemovedInSphinx30Warning, RemovedInSphinx40Warning
from spectrochempy import *
from docs import apigen

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx30Warning)
warnings.filterwarnings(action='ignore', category=RemovedInSphinx40Warning)
warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)

preferences = general_preferences
set_loglevel(WARNING)

SERVER = os.environ.get('SERVER_FOR_LCS', None)

PROJECT = "spectrochempy"
PROJECTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCESDIR = os.path.join(PROJECTDIR, "spectrochempy")
DOCDIR = os.path.join(PROJECTDIR, "docs")
USERDIR = os.path.join(PROJECTDIR, "docs", "user")
TUTORIALS = os.path.join(USERDIR, "tutorials", "*", "*.py")
USERGUIDE = os.path.join(USERDIR, "userguide", "*", "*.py")
API = os.path.join(DOCDIR, 'api', 'generated')
BUILDDIR = os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT)
DOCTREES = os.path.join(DOCDIR, '..', '..', '%s_doc' % PROJECT, '~doctrees')


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
        
        pipe = subprocess.Popen(  # -amend
            ["git", "commit", "--no-verify", "-m 'DOC:updated'", '%s' %
             OUTPUT],
            stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()


def api_gen(force=False):
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
                ],
                )


def make_docs(*args):
    """Make the html and pdf documentation

    Parameters
    ----------
    *args : tuple(any,...)
        Arguments among:
        
        * html
    """
    args = list(args)
    regenerate_api = False
    
    notebooks = False
    if 'notebooks' in args:
        notebooks = True
        # we need to use jupytext to sync py and ipynb files in userguide and tutorials
        cmds = (f"jupytext --sync {USERGUIDE}", f"jupytext --sync {TUTORIALS}")
        for cmd in cmds:
            res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = res.communicate()
            if not error:
                print(output.decode("utf-8"))
            else:
                print(error.decode("utf-8"))
    
    nocommit = True
    if 'commit' in args:
        nocommit = False
    
    # make sure commits have been done (if not nocommit flag set)
    if not nocommit:
        gitcommands()
    
    if 'DEBUG' in args:
        set_loglevel(DEBUG)
    
    builders = []
    if 'html' in args:
        builders.append('html')
    
    if 'pdf' in args:
        builders.append('latex')
    
    if 'clean' in args:
        clean()
        regenerate_api = True
        print('\nOld documentation now erased.\n')
    
    if builders:
        print('\nDocumentation directory are created.\n')
        make_dirs()
    
    if regenerate_api or not os.path.exists(API):
        api_gen(force=regenerate_api)
    
    for builder in builders:
        
        print('building %s documentation (version : %s)' % (builder, version))
        doc_version,sep = ('latest','/') if 'dev' in version else ('stable','/')
        
        srcdir = confdir = DOCDIR
        outdir = f"{BUILDDIR}/{builder}{sep}{doc_version}"
        doctreedir = f"{BUILDDIR}/~{doc_version}_doctrees"
        
        # with patch_docutils(), docutils_namespace():
        sp = Sphinx(srcdir, confdir, outdir, doctreedir, builder)
        sp.verbosity = 1
        
        sp.build()
        res = sp.statuscode
        debug_(res)
        
        if builder == 'latex':
            cmds = (f"cd {BUILDDIR}/latex",
                    "make",
                    f"mv {PROJECT}.pdf ../pdf/{PROJECT}.pdf")
            for cmd in cmds:
                res = subprocess.call(cmd, shell=True)
                print(res)
        
        if not nocommit:
            gitcommands()  # update repository
        
        print(f"\n{'-'*130}\nBuild finished. The {builder.upper()} pages are in {os.path.normpath(outdir)}.")
        
        # do some cleaning
        shutil.rmtree('auto_examples', ignore_errors=True)
    
    released = 'FALSE'
    if 'release' in args:
        update_html_page(outdir, doc_version)
        do_release()
        released = 'TRUE'
        
    
    print(f'\n\nReleased on spectrochempy.fr : {released}')
    if not notebooks:
        print('\nWARNING: Jupyter notebooks were not regenerated')
        print('if they are missing in the final documentation: use `notebooks` parameter! \n')
        
    print('-'*130)
    
    return True

def update_html_page(outdir, doc_version):
    """
    Modify page generated with sphinx (TODO: There is porbably a better method using sphinx templates to override
    the themes)
    """

    replace=f"""
<div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions">

    <span class="rst-current-version" data-toggle="rst-current-version">
      <span class="fa fa-book">SpectroChemPy</span>
      v: {doc_version}
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

"""+"""
<script type="text/javascript">
    jQuery(function () {
        SphinxRtdTheme.Navigation.enable(true);
    });
</script>
    """
    
    with open(os.path.join(outdir, 'index.html'), "r") as f:
        txt = f.read()
        
    regex = r"(<script type=\"text\/javascript\">.*SphinxRtdTheme.*script>)"
    result = re.sub(regex, replace, txt, 0, re.MULTILINE | re.DOTALL)
    
    if result:

        with open(os.path.join(outdir, 'index.html'), "w") as f:
            f.write(result)


def do_release():
    """Release/publish the documentation to the webpage.
    """
    
    # make the doc
    # make_docs(*args)
    
    # upload docs to the remote web server
    if SERVER:
        
        print("uploads to the server of the html/pdf files")
        path = sys.argv[0]
        while not path.endswith(PROJECT):
            path, _ = os.path.split(path)
        path, _ = os.path.split(path)
        FROM = os.path.join(path, '%s_doc/html' % PROJECT, '*')
        cmd = f'rsync -e ssh -avz  --exclude="~*" {FROM} {SERVER}:{PROJECT}/html/'
        print(cmd)
        debug_(subprocess.call(['pwd'], shell=True))  # , executable='/bin/bash'))
        
        res = subprocess.call([cmd], shell=True)  # , executable='/bin/bash')
        print(res)
        print('\n' + cmd + "Finished")
    
    else:
        error_('Cannot find the upload server : {}!'.format(SERVER))


    

def clean():
    """Clean/remove the built documentation.
    """
    
    shutil.rmtree(BUILDDIR + '/html', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/pdf', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/latex', ignore_errors=True)
    shutil.rmtree(BUILDDIR + '/~doctrees', ignore_errors=True)
    shutil.rmtree(BUILDDIR, ignore_errors=True)
    # shutil.rmtree(DOCDIR   + '/gen_modules', ignore_errors=True)
    # shutil.rmtree(DOCDIR   + '/gallery', ignore_errors=True)
    shutil.rmtree(API, ignore_errors=True)


def make_dirs():
    """Create the directories required to build the documentation.
    """
    
    # Create regular directories.
    build_dirs = [os.path.join(BUILDDIR, '~doctrees'),
                  os.path.join(BUILDDIR, 'html'),
                  os.path.join(BUILDDIR, 'latex'),
                  os.path.join(BUILDDIR, 'pdf'),
                  os.path.join(DOCDIR, '_static'),
                  ]
    for d in build_dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        # full make
        sys.argv.append('html')
    
    action = sys.argv[1]
    
    make_docs(*sys.argv[1:])
