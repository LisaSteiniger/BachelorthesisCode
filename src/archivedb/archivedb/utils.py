# -*- coding: utf-8 -*-
"""
    Small utility functions.
"""
import numpy as np
import datetime
from calendar import timegm


def to_timestamp(time_from_user, fmt=u"%Y-%m-%d %H:%M:%S"):
    """
        Converts a date time string to a UTC nanosecond time stamp.

        If a time stamp is given, i.e. an integer, does nothing and returns it back.

        Parameters
        ----------
        time_from_user : string
            Of the form 'YYYY-MM-DD HH:MM:SS.f' or 'YYYY-MM-DD HH:MM:SS'

        Returns
        -------
        out : int
            The corresponding UTC time stamp in nanoseconds.
    """
    if isinstance(time_from_user, (int, np.int64)):
        return time_from_user
    # checks?
    time_dateobj = datetime.datetime.strptime(time_from_user[:19], fmt)
    extra = 0
    if time_from_user[19:]:
        extra = int(float(time_from_user[19:])*1000000000)
    # from the format string it follows that we don't have fractions here
    return int(timegm(time_dateobj.timetuple())) * 1000000000 + extra


def to_stringdate(time_nano):
    """
        Converts a given UTC timestamp to a date string.

        Parameters
        ----------
        time_nano : int
            The time in UTC nanoseconds.
            Of the form: 1435601915999999999
        Returns
        -------
        to_timestamp : str
    """
    mytime = datetime.datetime.utcfromtimestamp(time_nano/1e9)
    return mytime.strftime('%Y-%m-%d %H:%M:%S.%f')
