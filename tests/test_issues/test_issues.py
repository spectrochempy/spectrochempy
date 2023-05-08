from spectrochempy.utils.testing import assert_approx_equal


def test_issue_643():

    import spectrochempy as scp

    # The following code works perfectly

    c0 = scp.LinearCoord.linspace(
        start=8000.0, stop=1250.0, num=6, labels=None, units="cm^-1", title="wavenumber"
    )

    ds = scp.NDDataset.full((6))
    ds.x = c0
    assert ds.x.data[0] == 8000.0
    assert ds.x.data[-1] == 1250.0

    ds.x.ito("nm")
    assert_approx_equal(ds.x.data[0], 1250.0)
    assert_approx_equal(ds.x.data[-1], 8000.0)

    # Importing data results in strange conversion

    ds2 = scp.read("wodger.spg")[0]
    assert_approx_equal(ds2.x.data[0], 5999, significant=4)

    ds2.x.ito("nm")
    assert_approx_equal(ds2.x.data[0], 1667, significant=4)
