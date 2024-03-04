# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Datetime utilities
"""

import re
import sys
from zoneinfo import ZoneInfo

if sys.version_info[1] < 12:
    from datetime import datetime
else:
    from datetime import datetime, UTC

import numpy as np

from spectrochempy.core.units import ur

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


def utcnow():
    """Return the current time in UTC with a timezone."""
    if sys.version_info[1] < 12:
        return datetime.utcnow().replace(microsecond=0, tzinfo=ZoneInfo("UTC"))
    else:
        return datetime.now(UTC).replace(microsecond=0)


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


def get_datetime_labels(data, resolution=None, labels=None):
    """
    A helper function to convert datetime axis to a relative time axis.

    Datetime are given in seconds (or other) from a acquisition date
    depending on the resolution of the datetimes. To change the default resolution,
    we can use the `resolution` parameter

    Parameters
    ----------
    data : an array of np.datetime64
        The data to be converted.
    resolution : str
        By default the data are in the units of the datetime object
        (often in seconds). To change this on can use one of the units among:
         * "days".
         * "hours".
         * "minute".
         * "second".
         * "millisecond".
         * "microsecond".
         * "nanosecond".
    labels : str, optional, default: None
        By default the axis label is given as a "relative time / <units>".
        If this parameter is set to "cf_format", then the axis label will include
        the acquisition date: "<units> since <acquisition_date>"

    Returns
    -------
    label : str
        The axis label
    data : numpy array of floats
        The array of values relative to the acquisition date.
    """
    data = np.asarray(data).ravel()
    acquisition_date = data[0]
    timedeltas = np.unique(np.diff(data))
    if resolution is None:
        for time_units in list(CF_TO_DT64_UNITS.keys()):
            if np.all(
                timedeltas / np.timedelta64(1, CF_TO_DT64_UNITS[time_units]) > 0.5
            ):
                break
    else:
        time_units = resolution

    if labels == "cf_format":
        label = f"{time_units} since {str(acquisition_date).replace('T',' ')}"
    else:
        units = from_dt64_units(CF_TO_DT64_UNITS[time_units])
        label = f"relative time / {units:~K}"
    newdata = (data - acquisition_date) / np.timedelta64(
        1, CF_TO_DT64_UNITS[time_units]
    )
    return label, newdata


def encode_datetime64(data, **attrs):
    label, data = get_datetime_labels(data, labels="cf_format")
    attrs["units"] = label
    attrs["calendar"] = "proleptic_gregorian"
    return data, attrs


def decode_datetime64(data, *attrs):
    """
    Utility to decode numpy.datetime64 encoded by encode_datetime64
    """
    raise NotImplementedError  # TODO: implement decode_datetime64


# Utility to convert between ISO8601 string, datetime, datetime64 and timestamps
# --------------------------------------------------------------------------------------
def strptime64(val, fmt=None, tz=None):
    # If created from a 64-bit integer, it represents an offset from
    # 1970-01-01T00:00:00.
    # If created from string, the string can be in ISO 8601 date or datetime
    # format.

    # Here we try to handle other case when it doesn't work.
    # Also we when date not NaT.

    def _parse(val):
        date = np.datetime64(val)
        if np.isnat(date):
            # we do not accept NaT in scpy
            raise ValueError  # pragma: no cover
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
            else:  # int(g[3]) > 31 (short year) or undefined (assume _axis_reversed)
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
        new_date = str(strptime64(date))

    return new_date


def windows_time_to_dt64(wt):
    import datetime

    base = datetime.datetime(1601, 1, 1, 0, 0, 0, 0)
    delta = datetime.timedelta(microseconds=wt / 10)
    return np.datetime64(base + delta)


if __name__ == "__main__":
    pass
