#! python3
# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================



import pytest
from glob import glob
import os, sys

from spectrochempy.utils.testing import example_run

@pytest.mark.parametrize('example', glob("../docs/user/**/*.py", recursive=True))
def test_example(example):

    name = os.path.basename(example)
    if (name in [__name__+'.py','conf.py','builddocs.py','apigen.py'] or
        'auto_examples' in example) :
            return

    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    if (name in ['tuto2_agir_IR_processing.py',
                 'tuto3_agir_tg_processing.py',
                 'agir_setup_figure.py',
                 '1_nmr.py',
                 '1_nmr-Copy1.py',
                 'fft.py',] ):
        print(example, ' ---> test skipped - DO IT MANUALLY')
        return

    print("testing", example)
    if os.path.exists(example) and os.path.splitext(example)[-1] == '.py':
        e, message, err = example_run(example)
        print(e, message.decode('utf8'), err )
        assert not e, message.decode('utf8')

