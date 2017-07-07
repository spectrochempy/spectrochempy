# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import os
import subprocess


# get the version string
# -----------------------
def get_version():
    """Get the version string

    Returns
    -------
    version: str
        the version string such as  |version|
    release: str
        the release string such as  |release|
    """

    version = '0.1'
    release = '0.1'

    try:

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "r") as f:
            version = f.readline()
            release = f.readline()

    except IOError:

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "w") as f:
            print(version, end='\n', file=f)
            print(release, end='\n', file=f)

    finally:
        pass

    try:
        # get the version number (if we are using git)
        verstr = subprocess.getoutput("git describe")
        verstr = verstr.split('-')

        # in case of a just tagged version version str contain only one string
        if len(verstr) >= 2:  # case of minor revision
            version = "%s.%s" % tuple(verstr[:2])
            release = verstr[0]
        else:
            version = verstr[0]
            release = version

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "w") as f:
            print(version, end='\n', file=f)
            print(release, end='\n', file=f)

    except:
        print('exception')

    finally:

        return version, release
