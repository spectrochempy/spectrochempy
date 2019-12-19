# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================



"""

"""

from spectrochempy.utils import Meta
from spectrochempy.utils.testing import  raises
from spectrochempy.units import ur

def test_init():
    meta = Meta()
    meta.td = [200, 400]
    assert meta.td[0] == 200
    assert meta.si is None
    meta['si'] = 'a string'
    assert isinstance(meta.si, str)
    assert meta.si.startswith('a')

def test_instance():
    meta = Meta()
    assert isinstance(meta, Meta)

def test_equal():
    meta1 = Meta()
    meta2 = Meta()
    assert meta1 == meta2

def test_readonly():
    meta = Meta()
    meta.chaine = "a string"
    assert meta.chaine == 'a string'
    meta.readonly = True
    with raises(ValueError):
        meta.chaine = "a modified string"
    assert meta.chaine != "a modified string"

def test_invalid_key():
    meta = Meta()
    meta.readonly = False   # this is accepted`
    with raises(KeyError):
        meta['readonly'] = True # this not because readonly is reserved
    with raises(KeyError):
        meta['_data'] = True # this not because _xxx type attributes are private

def test_get_inexistent():
    meta = Meta()
    assert meta.existepas is None

def test_get_keys_items():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 1024
    assert list(meta.keys()) == ['si', 'td']
    assert list(meta.items()) == [('si', 1024), ('td', [200, 400])]

def test_iterator():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024
    assert sorted([val for val in meta]) == ['ls', 'ns', 'si', 'td']

def test_copy():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024

    meta2 = meta
    assert meta2 is meta

    meta2 = meta.copy()
    assert meta2 is not meta
    assert sorted([val for val in meta2]) == ['ls', 'ns', 'si', 'td']

    # bug with quantity
    
    si = 2048 * ur.s
    meta.si = si

    meta3 = meta.copy()
    meta3.si = si / 2.
    
    assert meta3 is not meta
    

def test_swap():
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048
    meta.swap(1,2)
    assert meta.td == [200,500,400]
    assert meta.xe == [30, 80, 40]
    assert meta.si == 2048


def test_permute():
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048

    p = (2,0,1)
    meta.permute(*p)
    assert meta.td == [500,200,400]
    assert meta.xe == [80, 30, 40]
    assert meta.si == 2048




