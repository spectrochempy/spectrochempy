# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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


def test_api():
    import spectrochempy
    # test version
    from spectrochempy.api import APIref, log
    from spectrochempy import __version__ as version
    assert version.split('.')[0] == '0'
    assert version.split('.')[1][:1] == '1'
    # TODO: modify this for each release

    # test application

    print(('\n\nRunning : ', spectrochempy.api.running))
    assert version.startswith('0.1')

    log.warning('Ok, this is nicely executing!')

    assert 'np' in APIref
    assert 'NDDataset' in APIref
    assert 'abs' in APIref

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
    x = ip.run_cell('%%%%addscript -p project -o essai -n available_styles\n'
                    'print(available_styles())')

    print('result\n',x.result)
    assert x.result.strip() == 'Script essai created.'

