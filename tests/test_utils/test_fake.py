from spectrochempy.utils import generate_fake  # , show


def test_fake():

    nd, specs, concs = generate_fake()
    assert nd.shape == (50, 4000)

    # specs.plot()
    # concs.plot()
    # nd.plot()
    # show()
