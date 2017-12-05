# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



import os
import sys
import datetime
import subprocess
import setuptools_scm
from pkg_resources import get_distribution, DistributionNotFound

__all__ = ['get_version','get_version_date', 'get_release_date',
           'get_copyright']

# ............................................................................
def get_copyright():
    current_year = datetime.date.today().year
    copyright = '2014-{}'.format(current_year)
    copyright += ' - A.Travert and C.Fernandez @ LCS'
    return copyright

# .............................................................................
def get_version(root=os.path.join(os.path.dirname(__file__), '../..'),
                dist = 'spectrochempy'):

    try:
        # let's first try to get version from git
        dev_version = setuptools_scm.get_version(
                version_scheme='post-release',
                root=root,
                relative_to=__file__).split('+')[0]
    except:
        try:
            # let's try with the distribution version
            dev_version = get_distribution(dist).version

        except DistributionNotFound:
            from importlib import import_module
            version = import_module('version', dist+'.version')
            # this is a hack in order to be able to reuse this
            # function from different package
            #from spectrochempy.version import version
            dev_version = version.version

    path = os.path.join(root, dist, '__version__.py')

    with open(path, "w") as f:
        _v = dev_version.split('.post')
        if _v[0].endswith('.dev'):
            _v = _v[0].split('.dev')
        version = release = _v[0]
        if len(_v) > 1:
            version = "%s%d.dev"%(_v[0][:-1],int(_v[0][-1:])+1)
        f.write("version = '%s'\n" % version)
        # f.write("dev_version = '%s'\n" % dev_version) # finally we do
        # number the revision, because of the mess. any commit, leading to a
        #  new need for commit!
        f.write("release = '%s' " % release)

    return version, release, dev_version

# .............................................................................
def get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except:
        pass

# .............................................................................
def get_version_date():
    try:
        return subprocess.getoutput(
            "git log -1 --date='short' --format='%ad'")
    except:
        pass



# =============================================================================
# __main__
# =============================================================================
if __name__ == '__main__':

    print( get_version() )

