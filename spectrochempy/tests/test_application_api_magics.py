# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import pytest

from spectrochempy import application, APIref, log, version

application.set_loglevel('WARNING')
from spectrochempy import *


def test_api():

    assert 'EFA' in APIref
    assert 'CRITICAL' in APIref
    assert 'np' in APIref
    assert 'NDDataset' in APIref
    assert 'abs' in APIref

    # test version
    assert version.split('.')[0] == '0'
    assert version.split('.')[1][:1] == '1'
    assert version.startswith('0.1')

    # test log
    set_loglevel("WARNING")
    assert log.level == 30
    log.warning('Ok, this is nicely executing!')
    log.level=10
    assert log.level == 10  # DEBUG Level by default


def test_magic_addscript(ip):

    assert "available_styles" in ip.user_ns.keys()
    ip.run_cell("print(available_styles())", store_history=True)
    ip.run_cell("project = Project()", store_history=True)
    x = ip.magic('addscript -p project -o style -n available_styles 1')
                    # script with the definition of the function
                    # `available_styles` content of cell 2

    print("x", x)
    assert x.strip() == 'Script style created.'

    # with cell contents
    x = ip.run_cell('%%addscript -p project -o essai -n available_styles\n'
                    'print(available_styles())')

    print('result\n',x.result)
    assert x.result.strip() == 'Script essai created.'

def test_console(script_runner):
    # to test this, the scripts must be installed so the spectrochempy
    # package must be installed : use pip install -e .

    ret = script_runner.run('scpy')
    assert ret.success

