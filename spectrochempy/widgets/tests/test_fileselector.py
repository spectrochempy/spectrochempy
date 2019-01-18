from spectrochempy import *

def test_fileselector():

    fs = FileSelector(path = None, filters='spg')
    assert fs.path.endswith('tests/')
    assert fs.value.endswith('')

