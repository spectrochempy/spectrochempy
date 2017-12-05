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



import pytest
from glob import glob
import os, sys

from tests.utils import notebook_run, example_run, show_do_not_block


@pytest.mark.parametrize('notebook', glob("../docs/source/user/*/*.ipynb"))
@show_do_not_block
def test_notebooks(notebook):
    print(notebook)
    if '.ipynb_checkpoints' in notebook :
        return True
    if os.path.exists(notebook) and os.path.splitext(notebook)[
        -1] == '.ipynb' :
        nb, errors = notebook_run(notebook)
        assert errors == []

@pytest.mark.parametrize('example', glob("../docs/source/examples/*/*.py"))
@show_do_not_block
def test_example(example):
    if os.path.exists(example) and os.path.splitext(example)[-1] == '.py' :
        e, message, err = example_run(example)
        print(e, message.decode('ascii'), err )
        assert not e, message.decode('ascii')

