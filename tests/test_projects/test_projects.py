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

from spectrochempy.api import *
from spectrochempy.utils import SpectroChemPyWarning

import pytest
import os

from tests.utils import assert_array_equal

# Basic
# -------
def test_save_and_load_file_with_nofilename(IR_source_2D):

    A = IR_source_2D.copy()
    A.save()

    # no directory for saving passed ... it must be in data
    path = os.path.join(scpdata, A.id+'.scp')
    assert os.path.exists(path)

    # the filename should be stored in the object just saved
    assert A.filename == A.id+'.scp'
    assert A.directory == scpdata

    B = NDDataset.load(path)
    assert B.description == A.description
    assert_array_equal(A.data, B.data)

    # the filename should be stored in the object just loaded
    assert B.filename == A.filename
    assert B.directory == scpdata

    os.remove(path)

def test_project(ds1, ds2, dsm):

    myp = Project(name='AGIR processing', method='stack')

    ds1.name = 'toto'
    ds2.name = 'tata'
    dsm.name = 'titi'

    ds = ds1[:10,INPLACE]
    assert ds is ds1
    print(ds1.shape, ds.shape)

    myp.add_datasets(ds1, ds2, dsm)

    print(myp.datasets_names)
    assert myp.datasets_names[-1]=='titi'  # because toto has changed to *toto
    assert ds1.parent == myp


    # iteration
    d=[]
    for item in myp:
        d.append(item)

    assert d[1][0] == 'tata'

    ##
    # add sub project
    msp1 = Project(name='AGIR ATG')
    msp1.add_dataset(ds1)
    assert ds1.parent == msp1   #ds1 has changed of project
    assert ds1.name not in myp.datasets_names

    msp2 = Project(name='AGIR IR')

    myp.add_projects(msp1, msp2)

    print(myp)
    # an object can be accessed by it's name whatever it's type
    assert 'tata' in myp.allnames
    myp['titi']
    assert myp['titi'] == dsm

    # import multiple objects in Project
    myp2 = Project(msp1, msp2, ds1, ds2)  #multi dataset and project and no
    # names

    print(myp2)

    # Example from tutorial agir notebook
    proj =  Project(
            Project(name='P350', label=r'$\mathrm{M_P}\,(623\,K)$'),
            Project(name='A350', label=r'$\mathrm{M_A}\,(623\,K)$'),
            Project(name='B350', label=r'$\mathrm{M_B}\,(623\,K)$'),
            name='HIZECOKE_TEST' )

    assert proj.projects_names == ['A350', 'B350', 'P350']


    # add a dataset to a subproject
    ir = NDDataset([1,2,3])
    tg = NDDataset([1,3,4])
    proj.A350['IR'] = ir
    proj['TG'] = tg

    print(proj.A350)
    print(proj)
    print(proj.A350.label)

    proj.save('HIZECOKE_TEST')

    newproj = Project.load('HIZECOKE_TEST')

    print(newproj)

    assert str(newproj)==str(proj)
    assert newproj.A350.label == proj.A350.label

    #proj = Project.load('HIZECOKE')
    #assert proj.projects_names == ['A350', 'B350', 'P350']

    script_source = \
    'print("samples contained in the project are: %s"%proj.projects_names)'

    from spectrochempy.scripts.script import Script
    proj['print_info'] = Script('print_info',script_source)

    print(proj)

    proj.save()

    # execute
    run_script(proj.print_info, globals(), locals())
    proj.print_info.execute(globals(), locals())