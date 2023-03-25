# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Clean, build, and release the HTML and PDF documentation for SpectroChemPy.
```bash
  python make.py [options]
```
where optional parameters indicates which job(s) is(are) to perform.
"""

import argparse
import shutil
import sys
import warnings
import zipfile
from os import environ
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from sphinx.application import Sphinx
from sphinx.deprecation import RemovedInSphinx70Warning

from spectrochempy.api import preferences as prefs
from spectrochempy.api import version
from spectrochempy.utils.file import download_testdata
from spectrochempy.utils.system import sh
from apigen import Apigen

warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)
warnings.filterwarnings(action="ignore", category=RemovedInSphinx70Warning)

# CONSTANT
PROJECTNAME = "spectrochempy"
REPO_URI = f"spectrochempy/{PROJECTNAME}"
API_GITHUB_URL = "https://api.github.com"
URL_SCPY = "www.spectrochempy.fr"

# GENERAL PATHS
DOCS = Path(__file__).parent
TEMPLATES = DOCS / "_templates"
STATIC = DOCS / "_static"
PROJECT = DOCS.parent
DOCREPO = PROJECT / "build"
DOCTREES = DOCREPO / "~doctrees"
HTML = DOCREPO / "html"
LATEX = DOCREPO / "latex"
DOWNLOADS = HTML / "downloads"
SOURCES = PROJECT / PROJECTNAME

# DOCUMENTATION SRC PATH
SRC = DOCS
USERGUIDE = SRC / "userguide"
GETTINGSTARTED = SRC / "gettingstarted"
DEVGUIDE = SRC / "devguide"
REFERENCE = SRC / "userguide" / "reference"

# generated by sphinx
API = REFERENCE / "generated"
DEV_API = DEVGUIDE / "generated"
GALLERY = GETTINGSTARTED / "gallery"

# create the testdata directory

datadir = prefs.datadir
# this process is relatively long, so we do not want to do it several time:
download_testdata()  # (download done using mamba install spectrochempy_data)


# ======================================================================================
# Class BuildDocumentation
# ======================================================================================
class BuildDocumentation(object):
    def __init__(self):

        # determine if we are in the development branch (latest) or master (stable)
        if "+" in version:
            self._doc_version = "dirty"
        elif "dev" in version:
            print("\n\nWe are creating the latest (dev) documentation.\n")
            self._doc_version = "latest"
        else:
            print("\n\nWe are creating the stable documentation.\n")
            self._doc_version = "stable"

    def __call__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-H", "--html", help="create html pages", action="store_true"
        )

        parser.add_argument(
            "-P", "--pdf", help="create pdf manual", action="store_true"
        )

        parser.add_argument(
            "--clean",
            help="clean for a full regeneration of the documentation",
            action="store_true",
        )

        parser.add_argument(
            "--tutorials", help="zip and upload tutorials", action="store_true"
        )

        parser.add_argument(
            "--delnb", help="delete all ipynb files", action="store_true"
        )

        parser.add_argument(
            "--apigen",
            help="execute a full regeneration of the api",
            action="store_true",
        )

        parser.add_argument(
            "--norun",
            help="do not execute notebooks",
            action="store_true",
        )

        parser.add_argument("--all", help="Build all docs", action="store_true")

        args = parser.parse_args()

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            print("by default, option is set to --html")
            args.html = True

        self.norun = False
        if args.norun:
            self.norun = True

        if args.clean and (args.html or args.all):
            self.clean("html")

        if args.clean and (args.pdf or args.all):
            self.clean("latex")

        if args.clean or args.apigen:
            self.apigen()

        if args.delnb:
            self.delnb()

        if args.html:
            self.make_docs("html")

        if args.pdf:
            self.make_docs("latex")
            self.make_pdf()

        if args.all:
            self.make_docs("html")
            self.make_docs("latex")
            self.make_pdf()

        if args.tutorials:
            self.zip_tutorials()

    @property
    def doc_version(self):
        return self._doc_version

    @staticmethod
    def delnb():
        # Remove all ipynb before git commit
        for nb in SRC.rglob("**/*.ipynb"):
            sh.rm(nb)
        for nbch in SRC.glob("**/.ipynb_checkpoints"):
            sh(f"rm -r {nbch}")

    @staticmethod
    def make_redirection_page():
        # create an index page at the site root to redirect to the latest version.

        html = f"""
        <html>
        <head>
        <title>Redirect to the dev version of the documentation</title>
        <meta http-equiv="refresh" content="0; URL=latest">
        </head>
        <body>
        <p>
        We have moved away from the <strong>spectrochempy.github.io</strong> domain.
        If you're not automatically redirected, please visit us at
        <a href="https://{URL_SCPY}">https://{URL_SCPY}</a>.
        </p>
        </body>
        </html>
        """
        with open(HTML / "index.html", "w") as f:
            f.write(html)

    @staticmethod
    def _confirm(action):
        # private method to ask user to enter Y or N (case-insensitive).
        answer = ""
        while answer not in ["y", "n"]:
            answer = input(
                f"OK to continue `{action}` Y[es]/[N[o] ? ",
            ).lower()
        return answer[:1] == "y"

    def make_docs(self, builder="html"):
        # Make the html or latex documentation

        doc_version = self.doc_version

        if builder not in ["html", "latex"]:
            raise ValueError('Not a supported builder: Must be "html" or "latex"')

        BUILDDIR = DOCREPO / builder

        print(
            f'{"-" * 80}\n'
            f"building {builder.upper()} documentation ({doc_version.capitalize()} "
            f"version : {version})"
            f"\n in {BUILDDIR}"
            f'\n{"-" * 80}'
        )

        self.make_dirs()

        self.sync_notebooks()

        # run sphinx
        print(f"\n{builder.upper()} BUILDING:")
        srcdir = confdir = DOCS
        outdir = f"{BUILDDIR}/{doc_version}"
        doctreesdir = f"{DOCTREES}/{doc_version}"

        sp = Sphinx(str(srcdir), str(confdir), str(outdir), str(doctreesdir), builder)
        # verbosity: int = 0,
        # parallel: int = 0,
        # keep_going: bool = False,
        # pdb: bool = False
        sp.parallel = 8
        sp.verbosity = 1

        if self.norun:
            sp.config.nbsphinx_execute = "never"
            sp.config.plot_gallery = 0

        sp.build()

        print(
            f"\n{'-' * 130}\nBuild finished. The {builder.upper()} pages "
            f"are in {outdir}."
        )

        if doc_version == "stable":
            doc_version = "latest"
            # make also the latest identical
            sh(f"rm -rf {BUILDDIR}/latest")
            sh(f"cp -r  {BUILDDIR}/stable {BUILDDIR}/latest")

        if builder == "html":
            self.make_redirection_page()

        # a workaround to reduce the size of the image in the pdf document
        # TODO:probably better solution exists?
        if builder == "latex":
            self.resize_img(GALLERY, size=580.0)

    # pdf
    @staticmethod
    def resize_img(folder, size):
        # image resizing mainly for pdf doc

        for filename in folder.rglob("**/*.png"):
            image = imread(filename)
            h, l, c = image.shape
            ratio = 1.0
            if l > size:
                ratio = size / l
            if ratio < 1:
                # reduce size
                image_resized = resize(
                    image,
                    (int(image.shape[0] * ratio), int(image.shape[1] * ratio)),
                    anti_aliasing=True,
                )
                imsave(filename, (image_resized * 255.0).astype(np.uint8))

    def make_pdf(self):
        # Generate the PDF documentation

        doc_version = self.doc_version
        LATEXDIR = LATEX / doc_version
        print(
            "Started to build pdf from latex using make.... "
            "Wait until a new message appear (it is a long! compilation) "
        )

        print("FIRST COMPILATION:")
        sh(
            f"cd {LATEXDIR}; "
            f"lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex"
        )

        print("MAKEINDEX:")
        sh(f"cd {LATEXDIR}; makeindex spectrochempy.idx")

        print("SECOND COMPILATION:")
        sh(
            f"cd {LATEXDIR}; "
            f"lualatex -synctex=1 -interaction=nonstopmode spectrochempy.tex"
        )

        print("move pdf file in the download dir")
        sh(
            f"cp "
            f"{LATEXDIR / PROJECTNAME}.pdf {DOWNLOADS}/{doc_version}-{PROJECTNAME}.pdf"
        )

    @staticmethod
    def sync_notebooks():
        # Use  jupytext to sync py and ipynb files in userguide

        pyfiles = set()
        print(f'\n{"-" * 80}\nSync *.py and *.ipynb using jupytex\n{"-" * 80}')

        py = list(SRC.glob("**/*.py"))
        py.extend(list(SRC.glob("**/*.ipynb")))

        for f in py[:]:
            # do not consider some files
            if (
                "generated" in f.parts
                or ".ipynb_checkpoints" in f.parts
                or "gallery" in f.parts
                or "examples" in f.parts
                or "sphinxext" in f.parts
            ) or f.name in ["conf.py", "make.py"]:
                continue
            # add only the full path without suffix
            pyfiles.add(f.with_suffix(""))

        count = 0
        for item in pyfiles:

            py = item.with_suffix(".py")
            ipynb = item.with_suffix(".ipynb")

            args = None

            if not ipynb.exists():
                args = [
                    "--update-metadata",
                    '{"jupytext": {"notebook_metadata_filter":"all"}}',
                    "--to",
                    "ipynb",
                    py,
                ]

            elif not py.exists():
                args = [
                    "--update-metadata",
                    '{"jupytext": {"notebook_metadata_filter":"all"}}',
                    "--to",
                    "py:percent",
                    ipynb,
                ]

            if args is not None:
                print(f"sync: {item}")
                count += 1

                sh.jupytext(*args, silent=False)

        if count == 0:
            print("\nAll notebooks are up-to-date and synchronised with rst files")
        print("\n")

    def zip_tutorials(self):
        # make tutorials.zip
        print(f'\n{"-" * 80}\nMake tutorials.zip\n{"-" * 80}')

        # clean notebooks output
        for nb in DOCS.rglob("**/*.ipynb"):
            # This will erase all notebook output
            sh(
                f"jupyter nbconvert "
                f"--ClearOutputPreprocessor.enabled=True --inplace {nb}",
                silent=True,
            )

        # make zip of all ipynb
        def zipdir(path, dest, ziph):
            # ziph is zipfile handle
            for inb in path.rglob("**/*.ipynb"):
                if ".ipynb_checkpoints" in inb.parent.suffix:
                    continue
                basename = inb.stem
                sh(
                    f"jupyter nbconvert {inb} --to notebook"
                    f" --ClearOutputPreprocessor.enabled=True"
                    f" --stdout > out_{basename}.ipynb"
                )
                sh(f"rm {inb}", silent=True)
                sh(f"mv out_{basename}.ipynb {inb}", silent=True)
                arcnb = str(inb).replace(str(path), str(dest))
                ziph.write(inb, arcname=arcnb)

        zipf = zipfile.ZipFile("~notebooks.zip", "w", zipfile.ZIP_STORED)
        zipdir(SRC, "notebooks", zipf)
        zipdir(GALLERY / "auto_examples", Path("notebooks") / "examples", zipf)
        zipf.close()

        sh(
            f"mv ~notebooks.zip "
            f"{DOWNLOADS}/{self.doc_version}-{PROJECTNAME}-notebooks.zip"
        )

    def clean(self, builder):
        # Clean/remove the built documentation.

        print(f'\n{"-" * 80}\nCleaning\n{"-" * 80}')

        doc_version = self.doc_version

        if builder == "html":
            shutil.rmtree(HTML / doc_version, ignore_errors=True)
            print(f"removed {HTML / doc_version}")
            shutil.rmtree(DOCTREES / doc_version, ignore_errors=True)
            print(f"removed {DOCTREES / doc_version}")
            shutil.rmtree(API, ignore_errors=True)
            print(f"removed {API}")
            shutil.rmtree(DEV_API, ignore_errors=True)
            print(f"removed {DEV_API}")
            shutil.rmtree(GALLERY, ignore_errors=True)
            print(f"removed {GALLERY}")
        else:
            shutil.rmtree(LATEX / doc_version, ignore_errors=True)
            print(f"remove {LATEX / doc_version}")

        self.delnb()

    def make_dirs(self):
        # Create the directories required to build the documentation.

        doc_version = self.doc_version

        # Create regular directories.
        build_dirs = [
            DOCREPO,
            DOCTREES,
            HTML,
            LATEX,
            DOCTREES / doc_version,
            HTML / doc_version,
            LATEX / doc_version,
            DOWNLOADS,
        ]
        for d in build_dirs:
            Path.mkdir(d, parents=True, exist_ok=True)

    @staticmethod
    def apigen():
        print(f'\n{"-" * 80}\nRegenerate the reference API list\n{"-" * 80}')

        Apigen()


def main():
    environ["DOC_BUILDING"] = "yes"
    Build = BuildDocumentation()
    try:
        res = Build()
    except Exception:
        return 1

    del environ["DOC_BUILDING"]
    return res


# ======================================================================================
if __name__ == "__main__":
    sys.exit(main())
