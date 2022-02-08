# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
#

"""
Datetime utilities

"""
__all__ = [
    "strptime64",
    "get_datetime_labels",
    "decode_datetime64",
    "encode_datetime64",
    "to_dt64_units",
    "from_dt64_units",
    "to_utc_iso8601",
    "is_datetime64",
]


import numpy as np
import re

from spectrochempy.core.units import ur


def is_datetime64(obj):
    """
    Utility to check if obj or obj.data is a np.datetime64 object.

    Parameters
    ----------
    obj : np.ndarray like obj
        If obj implement NDDataset or Coord obj, then the data attribute is checked.

    Returns
    -------
    is_dt64 : bool
        Whether the obj or obj.data has a np.datetime64 dtype or not
    """
    dt64 = np.datetime64
    data = obj._data if hasattr(obj, "_data") else obj
    linear = obj._linear if hasattr(obj, "_linear") else False
    res = False
    if data is not None:
        res = isinstance(obj._data[0], dt64)
    elif linear:
        res = isinstance(obj._increment, dt64)
    return res


# Dicts and functions for the conversion between dt64 (numpy.datetime64) units
# to and from spectrochempy (pint) units
# --------------------------------------------------------------------------------------
DT64_TO_SCP_UNITS = {
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


def from_dt64_units(units):
    return ur.Unit(DT64_TO_SCP_UNITS[units])


def to_dt64_units(units):
    dt64_units = {v: k for k, v in DT64_TO_SCP_UNITS.items()}
    return dt64_units[str(units)]


# Dict and function for the conversion of CF (http://cfconventions.org) to dt64 units
# --------------------------------------------------------------------------------------
CF_TO_DT64_UNITS = {
    "days": "D",
    "hours": "h",
    "minutes": "m",
    "seconds": "s",
    "milliseconds": "ms",
    "microseconds": "us",
    "nanoseconds": "ns",
}


def get_datetime_labels(data, resolution=None):
    """
    A helper function to convert datetime axis to the CF format

    Datetime are given in seconds (or other) from a reference date
    depending on the resolution of the datetimes. To change the default resolution,
    we can use the `resolution` parameter

    Parameters
    ----------
    data : an array of np.datetime64
        The data to be converted.
    resolution : str
        By default the data are in the units of the CF object
        (often in seconds). To change this on can use one of the units among:

         * "days".
         * "hours".
         * "minute".
         * "second".
         * "millisecond".
         * "microsecond".
         * "nanosecond".

    Returns
    -------
    label : str
        A label in the form "<resolution> since <reference_date>"
    data : an array of float
        The array values relative to the reference date.
    """
    data = np.asarray(data).ravel()
    reference_date = data[0]
    timedeltas = np.unique(np.diff(data))
    zero = np.timedelta64(0, "ns")
    if resolution is None:
        for time_units in CF_TO_DT64_UNITS.keys():
            if np.all(
                timedeltas % np.timedelta64(1, CF_TO_DT64_UNITS[time_units]) == zero
            ):
                break
    else:
        time_units = resolution

    label = f"{time_units} since {str(reference_date).replace('T',' ')}"
    newdata = (data - reference_date) / np.timedelta64(1, CF_TO_DT64_UNITS[time_units])
    return label, newdata


def encode_datetime64(data, **attrs):
    label, data = get_datetime_labels(data)
    attrs["units"] = label
    attrs["calendar"] = "proleptic_gregorian"
    return data, attrs


def decode_datetime64(data, *attrs):
    """
    Utility to decode numpy.datetime64 encoded by encode_datetime64
    """
    raise NotImplementedError  # TODO: implement decode_datetime64


# Utility to convert between ISO8601 string, datetime, datetime64 and timestamps
# ----------------------------------------------------------------------
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


def to_utc_iso8601(date):

    if isinstance(date, np.datetime64):
        new_date = str(date)  # directly in a good ISO format  (UTC)

    elif isinstance(date, str):
        # format YYYY-MM-DD[T]HH:MM:SS.ffffff (higher resolution: us)
        # assume datetime are given in local time. They stored in UTC using the timezone
        # dataset attribute if set of the local timezone of the platform where
        # spectrochempy is executed.

        new_date = np.datetime64(date)

    return new_date


if __name__ == "__main__":

    from datetime import datetime

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

    dt = datetime.utcnow()
    dt64 = np.datetime64(dt)
    ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
    assert datetime.utcfromtimestamp(ts) == dt

    print(to_utc_iso8601(dt64))  # convert from datetime64
    print(to_utc_iso8601("2022-02-04 09:09:46"))
