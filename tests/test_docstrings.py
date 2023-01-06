import spectrochempy
from spectrochempy.utils import check_docstrings as td


def test_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.ndarray"
    result = td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.ndarray.NDArray,
        exclude=[
            "SA01",
            "EX01",
            # temporary exclude some checks
            "GL08",  # The object does not have a docstring"
            "ES01",  # Extended summary missing
            "PR01",  # Parameters {'**kwargs'} not documented
            "RT01",  # No Returns section found
            "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned"
            "GL11",  # Other Parameters section missing while `**kwargs` is in class or method signature."
        ],
    )
    print(result)
