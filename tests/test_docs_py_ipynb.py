# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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

import pytest
from glob import glob
import os, sys

from tests.utils import notebook_run, example_run, show_do_not_block

@pytest.mark.parametrize('notebook', glob("../docs/source/userguide/*.ipynb"))
@show_do_not_block
def test_notebooks(notebook):
    if os.path.exists(notebook) and os.path.splitext(notebook)[-1] != '.ipynb':
        nb, errors = notebook_run(notebook)
        assert errors == []

@pytest.mark.parametrize('example', glob("../docs/source/examples/*/*.py"))
@show_do_not_block
def test_example(example):
    if os.path.exists(example) and os.path.splitext(example)[-1]!='.py':
        e, message, err = example_run(example)
        print(e, message.decode('ascii'), err )
        assert not e, message.decode('ascii')

