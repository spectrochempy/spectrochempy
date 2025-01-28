# DOCUMENTATION Readme

This directory contains the sources ans tools necessary to build the spectrochempy
documentation using Sphinx.

To generate the documentation use:

  python docs/make.py html

Note that changelogs are updated only if prior to calling make.py, you first
execute:

   python .github/workflows/scripts/update_version_and_release_notes.py

The html pages are  in build/html/latest.  Just open build/html/latest/index.html in a browser to see this pages.
