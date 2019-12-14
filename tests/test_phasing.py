from spectrochempy import *

from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         raises, catch_warnings,
                                         assert_approx_equal)

def test_nmr_manual_1D_phasing():
  
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.*ur.Hz)           # inplace broadening
    transf = ndd.fft(tdeff=8192, size=2**15)  # fft
    transf.plot()  # plot)
    
    # manual phasing
    transfph = transf.pk(verbose=True)   # by default pivot = 'auto'
    transfph.plot(xlim=(20,-20), clear=False, color='r')
    assert_array_equal(transfph.data,transf.data)         # because phc0 already applied
    
    transfph3 = transf.pk(pivot=50 , verbose=True)
    transfph3.plot(clear=False, color='r')
    not assert_array_equal(transfph3.data,transfph.data)         # because phc0 already applied
    #
    transfph4 = transf.pk(pivot=100, phc0=40., verbose=True)
    transfph4.plot(xlim=(20,-20), clear=False, color='g')
    assert transfph4 != transfph

    transfph4 = transf.pk(pivot=100, verbose=True, inplace=True)
    (transfph4-10).plot(xlim=(20,-20), clear=False, color='r')
    
    show()

def test_nmr_auto_1D_phasing():
    
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.*ur.Hz)           # inplace broadening

    transf = ndd.fft(tdeff=8192, size=2**15)
    transf.plot(xlim=(20,-20), ls=':', color='k')

    transfph2 = transf.pk(verbose=True)
    transfph2.plot(xlim=(20,-20), clear=False, color='r')
    
    # automatic phasing
    transfph3 = transf.apk(verbose=True)
    (transfph3-1).plot(xlim=(20,-20), clear=False, color='b')

    transfph4 = transf.apk(algorithm='acme', verbose=True)
    (transfph4-2).plot(xlim=(20,-20), clear=False, color='g')

    transfph5 = transf.apk(algorithm='neg_peak', verbose=True)
    (transfph5-3).plot(xlim=(20,-20), clear=False, ls='-', color='r')

    transfph6 = transf.apk(algorithm='neg_area', verbose=True)
    (transfph6-4).plot(xlim=(20,-20), clear=False, ls='-.', color='m')

    transfph4 = transfph6.apk(algorithm='acme', verbose=True)
    (transfph4-6).plot(xlim=(20,-20), clear=False, color='b')

    show()

def test_nmr_multiple_manual_1D_phasing():
    
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.*ur.Hz)           # inplace broadening
    
    transf = ndd.fft(tdeff=8192, size=2**15)
    
    transfph1 = transf.pk(verbose=True)
    transfph1.plot(xlim=(20,-20), color='k')

    transfph2 = transf.pk(verbose=True)
    transfph2.plot(xlim=(20,-20), clear=False, color='r')

    transfph3 = transf.pk(52.43836, -16.8366, verbose=True)
    transfph3.plot(xlim=(20,-20), clear=False, color='b')
    
    show()

def test_nmr_multiple_auto_1D_phasing():
    
    path = os.path.join(general_preferences.datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd /= ndd.real.data.max()  # normalize
    ndd.em(10.*ur.Hz)           # inplace broadening
    
    transf = ndd.fft(tdeff=8192, size=2**15)
    transf.plot(xlim=(20,-20), ls=':', color='k')
    
    t1 = transf.apk(algorithm='neg_peak',verbose=True)
    (t1-5.).plot(xlim=(20,-20), clear=False, color='b')
    
    t2 = t1.apk(algorithm='neg_area', verbose=True)
    (t2-10).plot(xlim=(20,-20), clear=False, ls='-.', color='m')
    
    t3 = t2.apk(algorithm='acme', verbose=True)
    (t3-15).plot(xlim=(20,-20), clear=False, color='r')
    
    show()