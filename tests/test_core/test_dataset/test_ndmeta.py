# -*- coding: utf-8 -*-
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

"""

"""

from spectrochempy.api import Meta
from spectrochempy.utils import SpectroChemPyWarning
from tests.utils import  raises

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

def test_swapaxes():
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048
    meta.swapaxes(1,2)
    assert meta.xe == [30,80,40]



