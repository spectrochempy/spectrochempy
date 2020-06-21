# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import sys
import os
import re
import requests

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

os.environ['USE_TQDM'] = 'No'
from spectrochempy import version

class UpdateWindow(QMainWindow):

    def __init__(self):

        new_version = self.check_for_updates()

        if new_version:

            message = f"""\
You are running SpectrocChemPy-{version}
but version {new_version} is available.
    
Please consider updating for bug fixes and new features !

Use: 
conda update {'-c spectrocat/label/dev ' if  'dev' in new_version else ''}spectrochempy"""
            QMainWindow.__init__(self)
            QMessageBox.warning(self, "SpectroChempy has been updated", message)


    def check_for_updates(self):
        # Gets version
        conda_url = "https://anaconda.org/spectrocat/spectrochempy/files"
        try:
            response = requests.get(conda_url)
        except requests.exceptions.RequestException:
            return None

        regex = r"\/(\d{1,2})\.(\d{1,2})\.(\d{1,2})\/download\/noarch" \
                r"\/spectrochempy-\d{1,2}\.\d{1,2}\.\d{1,2}\-(dev\d{1,2}|stable).tar.bz2"
        matches = re.finditer(regex, response.text, re.MULTILINE)
        vavailables = {}
        for matchNum, match in enumerate(matches):
            vavailables[match[0]] = (match[1], match[2], match[3], match[4])

        rel = version.split('+')[0]
        rel = list(rel.split('.'))
        major, minor, patch = map(int, rel[:3])
        dev = 0 if len(rel) == 3 else int(rel[3][3:])

        # check the online version string
        new_major, new_minor, new_patch, new_dev = major, minor, patch, dev
        upd = False
        dev_upd=False
        #print(major, minor, patch, dev)
        for k, v in vavailables.items():
            _major, _minor, _patch = list(map(int, v[:3]))
            _dev = int(v[3][3:]) if v[3] is not None and 'dev' in v[3] else 0
            #print (_major, _minor, _patch, _dev)
            if _major > new_major:
                new_major = _major
                new_dev = 0  # reset dev
                upd = True
            elif _minor > new_minor:
                new_minor = _minor
                new_dev = 0  # reset dev
                upd = True
            elif _patch > new_patch:
                new_patch = _patch
                new_dev = 0  # reset dev
                upd = True

            # check dev
            if _dev > new_dev and _major >= new_major and _minor >= new_minor and _patch >= new_patch:
                new_dev = _dev
                upd = True
                dev_upd = True

        new_version = False
        if upd:
            new_version = f'{new_major}.{new_minor}.{new_patch}'
            if dev_upd:
                new_version = f'{new_version}.dev{new_dev}'

        return new_version

def main():
    _ = QApplication(sys.argv)
    UpdateWindow()
    sys.exit(0)


if __name__ == '__main__':
    main()
