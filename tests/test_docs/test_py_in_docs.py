# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# Testing examples and notebooks (Py version) in docs
# ----------------------------------------------------------------------------------------------------------------------

import os
from glob import glob

import pytest

path = os.getcwd()
if "/tests" in path:
    path = path[:path.find('/tests')]


# ......................................................................................................................
def example_run(path):
    import subprocess

    pipe = None
    so=None
    serr=None
    try:
        pipe = subprocess.Popen(
                ["python", path, '--nodisplay'],
                stdout=subprocess.PIPE)
        (so, serr) = pipe.communicate()
    except Exception:
        pass

    return pipe.returncode, so, serr

# ......................................................................................................................
@pytest.mark.parametrize('example', glob(os.path.join(path, 'docs', 'user', '**', '*.py'), recursive=True))
def test_example(example):
    name = os.path.basename(example)
    if (name in [__name__ + '.py', 'conf.py', 'builddocs.py', 'apigen.py'] or 'auto_examples' in example):
        return

    # some test will failed due to the magic commands or for other known reasons
    # SKIP THEM
    if (name in ['tuto2_agir_IR_processing.py',
                 'tuto3_agir_tg_processing.py',
                 'agir_setup_figure.py',
                 '1_nmr.py',
                 '1_nmr-Copy1.py',
                 'fft.py',
                 'Import.py']):
        print(example, ' ---> test skipped - DO IT MANUALLY')
        return

    if os.path.exists(example) and os.path.splitext(example)[-1] == '.py':
        e, message, err = example_run(example)
        print(e, message.decode('utf8'), err)
        assert not e, message.decode('utf8')
