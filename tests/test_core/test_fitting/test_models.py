import numpy as np

import spectrochempy as scp
from spectrochempy.units import ur
from spectrochempy.utils.testing import assert_approx_equal
import matplotlib.pyplot as plt


def test_models():

    model = scp.asymmetricvoigtmodel()
    assert model.args == ["ampl", "pos", "width", "ratio", "asym"]

    x = np.arange(1000)
    ampl = 1000.0
    width = 100
    ratio = 0
    asym = 1.5
    pos = 500

    max = 9.3944

    array = model.f(x, ampl, pos, width, ratio, asym)
    assert array.shape == (1000,)
    assert_approx_equal(array[pos], max, significant=4)

    array = model.f(x, 2.0 * ampl, pos, width, ratio, asym)  # ampl=2.
    assert_approx_equal(array[pos], max * 2.0, significant=4)

    # x array with units
    x1 = x * ur("cm")
    array = model.f(x1, ampl, pos, width, ratio, asym)
    assert_approx_equal(array[pos], max, significant=4)
    assert not hasattr(array, "units")

    # amplitude with units
    ampl = 1000.0 * ur("g")
    array = model.f(x1, ampl, pos, width, ratio, asym)
    assert hasattr(array, "units")
    assert array.units == ur("g")
    assert_approx_equal(array[pos].m, max, significant=4)

    # use keyword instead of positional parameters
    array = model.f(x1, ampl, pos, asym=asym, width=width, ratio=ratio)
    assert_approx_equal(array[pos].m, max, significant=4)

    # rescale some parameters
    array = model.f(
        x1, width=1000.0 * ur("mm"), ratio=ratio, asym=asym, ampl=ampl, pos=pos
    )
    assert_approx_equal(array[pos].m, max, significant=4)

    # x is a Coord object
    x2 = scp.LinearCoord.arange(1000)
    width = 100.0
    array = model.f(x2, ampl, pos, width, ratio, asym)
    assert isinstance(array, scp.NDDataset)
    assert_approx_equal(array[pos].value.m, max, significant=4)
    assert array.units == ampl.units

    # x is a Coord object with units
    x3 = scp.LinearCoord.linspace(0.0, 0.999, 1000, units="m", title="distance")
    width = 100.0 * ur("mm")
    pos = 0.5
    array = model.f(x3, ampl, pos, width, ratio, asym)
    assert hasattr(array, "units")
    assert_approx_equal(array[500].m, max, significant=4)

    # do the same for various models
    kwargs = dict(
        ampl=1.0 * ur["g"],
        width=100.0 * ur("mm"),
        ratio=0.5,
        asym=2,
        pos=0.5,
        c_2=1.0,
    )

    for modelname, expected in [
        ("gaussianmodel", 0.9394292818892936),
        ("lorentzianmodel", 0.6366197723675814),
        ("voigtmodel", 0.14024341343686939),
        ("asymmetricvoigtmodel", 0.7880285255336164),
        ("polynomialbaseline", 0.0),
        ("sigmoidmodel", 50),
    ]:
        print("\nmodel:", modelname)
        model = getattr(scp, modelname)()
        if modelname == "sigmoid":
            kwargs["width"] = 0.01
        array = model.f(x3, **kwargs)
        actual = array[pos].value
        if modelname != "sigmoid":
            actual = actual * 100
        assert_approx_equal(actual.m, expected, 4)
        array.plot(title=modelname)
        plt.show()
