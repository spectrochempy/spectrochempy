# -*- coding: utf-8 -*-

#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================
#

"""
Datetime utilities

"""
__all__ = ["strptime64", "encode_datetime64", "to_dt64_units", "from_dt64_units"]


import numpy as np
import re

from spectrochempy.core.units import ur

DT64_UNITS = {
    "Y": "year",
    "M": "month",
    "W": "week",
    "D": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
    "ps": "picosecond",
    "fs": "femtosecond",
    "as": "attosecond",
}


def strptime64(val, fmt=None):

    # If created from a 64-bit integer, it represents an offset from
    # 1970-01-01T00:00:00.
    # If created from string, the string can be in ISO 8601 date or datetime
    # format.

    # Here we try to handle other case when it doesn't work.
    # Also we when date not NaT.

    def _parse(val):
        date = np.datetime64(val)
        if np.isnat(date):  # we do not accept NaT in scpy
            raise ValueError
        return date

    def _mysubst(match):
        g = match.groups()

        if g[0] is None and g[4] is None:
            return None

        if g[0] is not None:  # date group present
            # YEAR?
            if int(g[1]) > 99:  # year (long) in first
                # positions
                year = g[1]
                reversed = False
            elif int(g[3]) > 99:
                year = g[3]
                # days = g[1]
                reversed = True
            elif int(g[1]) > 31:  # short year
                siecle = "20" if int(g[1]) < 70 else "19"
                year = f"{siecle}{g[1]}"
                reversed = False
            else:  # int(g[3]) > 31 (short year) or undefined (assume reversed)
                siecle = "20" if int(g[3]) < 70 else "19"
                year = f"{siecle}{g[3]}"
                reversed = True

            # MONTH and DAYS
            month = g[2]
            if reversed:
                # assume days first
                days = g[1]
            else:
                days = g[3]

            if int(month) > 12:
                # nope days and month are inversed
                days, month = month, days

            date = f"{year}-{month}-{days}"

        else:
            date = "1970-01-01"  # base date

        time = f"T{g[4]}" if g[4] is not None else ""

        val = f"{date}{time}"

        return val

    def _regex_parse(val):
        regex = (
            r"^((\d{2,4})[\/\-\.](\d{2})[\/\-\.](\d{2,4}))?\s?(\d{2}\:\d{2}\:\d{2})?"
        )

        val = re.sub(regex, _mysubst, val, 0)
        date = _parse(val)
        if np.isnat(date):
            raise ValueError
        else:
            return date

    try:
        date = _parse(val)
    except ValueError:
        try:
            date = _regex_parse(val)
        except ValueError:
            raise

    return date.astype("datetime64[us]")


def from_dt64_units(units):
    return ur.Unit(DT64_UNITS[units])


def to_dt64_units(units):
    dt64_units = {v: k for k, v in DT64_UNITS.items()}
    return dt64_units[str(units)]


def encode_datetime64(data, attrs={}):

    data = np.asarray(data).ravel()
    reference_date = data[0]

    timedeltas = np.unique(np.diff(data))
    zero = np.timedelta64(0, "ns")
    cf_to_dt64_units = {
        "days": "D",
        "hours": "h",
        "minutes": "m",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
        "nanoseconds": "ns",
    }
    for time_unit in cf_to_dt64_units.keys():
        if np.all(timedeltas % np.timedelta64(1, cf_to_dt64_units[time_unit]) == zero):
            break
    attrs["units"] = f"{time_unit} since {reference_date}"
    attrs["calendar"] = "proleptic_gregorian"

    data = (data - reference_date) // np.timedelta64(1, cf_to_dt64_units[time_unit])

    return data, attrs
