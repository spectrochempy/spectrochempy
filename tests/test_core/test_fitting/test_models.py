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
    asym = 0.0
    pos = 500

    max = 9.3944

    array = model.f(x, ampl, pos, width, ratio, asym)
    assert array.shape == (1000,)
    assert_approx_equal(array[pos], max, significant=4)
    plt.plot(array)
    plt.show()

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
    assert_approx_equal(array[0].m, max, significant=4)

    # use keyword instead of positional parameters
    array = model.f(x1, width=width, ratio=ratio, asym=asym, ampl=ampl, pos=pos)
    assert_approx_equal(array[0].m, 0.9394, significant=4)

    # rescale some parameters
    array = model.f(x1, width=10 * ur("mm"), ratio=ratio, asym=asym, ampl=ampl, pos=pos)
    assert_approx_equal(array[0].m, 0.9394, significant=4)

    # x is a Coord object
    x2 = scp.LinearCoord.arange(10)
    width = 1.0
    array = model.f(x2, ampl, pos, width, ratio, asym)
    assert hasattr(array, "units")
    assert_approx_equal(array[0].m, 0.9394, significant=4)

    # x is a Coord object with units
    x3 = scp.LinearCoord.arange(10, units="m")
    width = 1000.0 * ur("mm")
    array = model.f(x3, ampl, pos, width, ratio, asym)
    assert hasattr(array, "units")
    assert_approx_equal(array[0].m, 0.9394, significant=4)

    # do the same for various models
    kwargs = dict(
        ampl=1.0 * ur["s"], width=1.0, ratio=0.5, asym=0.5, pos=0.0, c_3=1.0, c_4=0.0
    )

    for modelname, expected in [
        ("gaussianmodel", 0.9394292818892936),
        ("lorentzianmodel", 0.6366197723675814),
        ("voigtmodel", 0.8982186579508358),
        ("asymmetricvoigtmodel", 0.7880285255336164),
        ("polynomialbaseline", 0.0),
        # ("sigmoid", 0.0)
    ]:
        model = getattr(scp, modelname)()
        array = model.f(x2, **kwargs)
        assert_approx_equal(array[0].m, expected, 4)
