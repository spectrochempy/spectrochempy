#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import numpy as np
from datetime import datetime

from spectrochempy.utils.datetime64 import strptime64


def test_get_strptime64_method():

    ref_date = np.datetime64("1970-01-11T00:00:00").astype("datetime64[us]")
    assert ref_date.dtype == np.dtype("datetime64[us]")

    date = strptime64("1970-01-11 00:00:00")
    assert date == ref_date

    date = strptime64("1970-01-11T00:00:00+02:00")
    assert date == ref_date - np.timedelta64(2, "h")

    date = strptime64(datetime(1990, 1, 1))
    assert date.astype("datetime64[Y]") == ref_date.astype(
        "datetime64[Y]"
    ) + np.timedelta64(20, "Y")

    date = strptime64("1970/01/11")
    # This cannot be parsed directly
    assert date == ref_date

    date = strptime64("1970-01-11")
    assert date == ref_date

    date = strptime64("1970.01.11")
    assert date == ref_date

    date = strptime64("11.01.70")
    assert date == ref_date

    date = strptime64("70.01.11")
    assert date == ref_date

    date = strptime64("16.01.15")  # ambiguous!
    assert date == np.datetime64("2015-01-16")

    date = strptime64("11.24.16")  # month first
    assert date == np.datetime64("2016-11-24")

    time = strptime64("11:12:59")
    assert time.astype(int) // 1000000 // 3600 == 11

    # date = strptime64("Fri Aug 30 09-35-07 2013")
