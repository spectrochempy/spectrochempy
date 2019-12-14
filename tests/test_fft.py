from spectrochempy import *

def test_1D_fft():
  
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.*ur.Hz)           # inplace broadening
    transf = ndd.fft(tdeff=8192, size=2**15)  # fft
    transf.plot(xlim=(20,-20), clear=False, color='r')
    
    show()

def test_1D_fft_out_hz():
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    LB = 10.*ur.Hz
    GB = 50.*ur.Hz
    ndd.gm(gb=GB, lb=LB)
    transf1 = ndd.fft(size=32000, ppm=False)
    _ = transf1.plot(xlim=[5000,-5000])
    
    show()