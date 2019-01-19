from spectrochempy import *

def test_fileselector():

    path = general_preferences.datadir
    fs = FileSelector(path = path, filters='spg')

    assert fs.path.endswith('testdata/')
    if fs.value is not None:
        assert fs.value.endswith('')

