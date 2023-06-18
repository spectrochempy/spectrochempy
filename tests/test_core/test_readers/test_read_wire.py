import spectrochempy as scp


def test_read_wire():
    # First read a single spectrum (measurement type : single)
    dataset = scp.read_wire("ramandata/wire/sp.wdf")
    _ = dataset.plot()

    # Now read a series of spectra (measurement type : series) from a Z-depth scan.
    dataset = scp.read_wire("ramandata/wire/depth.wdf")
    _ = dataset.plot_image()

    # extract a line scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/line.wdf")
    _ = dataset.plot_image()

    # finally extract grid scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/mapping.wdf")
    _ = dataset.sum(dim=2).plot_image()

    # show spectra if test run as a single pytest test
    scp.show()
