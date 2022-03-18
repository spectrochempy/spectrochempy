from spectrochempy.utils.orderedset import OrderedSet


def test_orderedset():
    s = OrderedSet("abracadaba")
    t = OrderedSet("simsalabim")
    assert s | t == OrderedSet(["a", "b", "r", "c", "d", "s", "i", "m", "l"])
    assert s & t == OrderedSet(["a", "b"])
    assert s - t == OrderedSet(["r", "c", "d"])
