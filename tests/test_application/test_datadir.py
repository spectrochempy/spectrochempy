def test_datadir():
    # test print a listing of the testdata directory
    from spectrochempy.application import DataDir

    print(DataDir().listing())
    # or simply
    print(DataDir())
    assert str(DataDir()).startswith("testdata")
