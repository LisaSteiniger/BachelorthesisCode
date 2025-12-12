# -*- coding: utf-8 -*-
"""
    Functions to read time interval information.
"""
import numpy as np
import datetime
import re
import logging
from . import utils
from . import url as _url
from . import programs
from . import cache

_re_from = re.compile("from=([0-9]{19})")
_re_upto = re.compile("upto=([0-9]{19})")


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False)
# settings={"useMemCache": False,
# "maxCacheLength" : 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="intervals",
                           name_suffix=".npz")
def get_time_intervals(signal_name, time_from, time_to, maxIntervals=None,
                       timeout=1, protocol="json",
                       cacheSettings=None, useCache=True):
    """
        Return time intervals where data samples exist.

        Intervals are ordered from the newest to the oldest.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        time_from, time_to: int or string
            Time boundaries either as a string of the format
            'YYYY-MM-DD HH:MM:SS.%f' or a nanosecond time stamp.
        maxIntervals : int, optional
            Return at most this number of intervals. If None all 
            intervals are returned.
        timeout, protocol are passed to read url functions.

        Returns
        -------
        time_intervals : nd.array
            2d array of time intervals.
            Each element of the list contains a 2-integer element list
            indicating the upper and lower end of the interval.
        See Also
        --------
        get_latest_time_interval : Return latest valid time interval.

        Examples
        --------
        >>> get_time_intervals('Test/raw/W7X/QSB_...',
                '2015-05-29 00:00:00','2015-06-30 00:00:00')
    """
    logger = logging.getLogger(__name__)
    logger.debug("find time intervals")
    baseaddress = _url._make_string_signal_name(signal_name)

    time_from = utils.to_timestamp(time_from)
    if time_to is None:
        logger.debug("no top boundary supplied, use current time")
        time_to = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    time_to = utils.to_timestamp(time_to)

    address = (baseaddress+'?filterstart=' + str(time_from)
               + '&filterstop=' + str(time_to))
    timestamp_intervals = []
    # keep reading pages and following the next page links (if available)
    while address is not None:
        d = _url.read_from_url_and_decode(address, timeout=timeout,
                                          protocol=protocol)
        urllinks = d.get("_links", {}).get("children", [])
        for link in urllinks:
            url = link.get("href", "")
            match1 = _re_from.search(url)
            match2 = _re_upto.search(url)
            if match1 is None or match2 is None:
                continue
            # note, group 0 is the entire string
            timestamp_intervals.append([int(match1.group(1)),
                                        int(match2.group(1))])
            if (maxIntervals is not None and
                    len(timestamp_intervals) >= maxIntervals):
                return np.array(timestamp_intervals)
        address = d.get("_links", {}).get("next", {}).get("href", None)
    return np.array(timestamp_intervals)


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False)
# settings={"useMemCache": False,
# "maxCacheLength" : 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="_lastinterval",
                           name_suffix=".npz")
def get_latest_time_interval(signal_name, timeout=1, protocol="json",
                             cacheSettings=None, useCache=True):
    """
        Return latest valid time stamp interval where data samples exist.

        Parameters
        ----------
        signal_name : string or list
            The signal address.

        Returns
        -------
        get_latest_time_interval : ndarray
            The last time intervals with data samples.

        See Also
        --------
        get_time_intervals : Return valid time stamp intervals.

        Examples
        --------
        >>> get_latest_time_interval('ArchiveDB/raw/W7X/CoDaStationDesc...')
    """
    return get_time_intervals(signal_name, "2010-01-01 00:00:01",
                              None, maxIntervals=1, timeout=timeout,
                              protocol=protocol,
                              useCache=useCache,
                              cacheSettings=cacheSettings).squeeze()


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False,
                           hash_keywords=["tstart", "tstop"])
# settings={"useMemCache": False,
# "maxCacheLength" : 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.INTERVALS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="intervals",
                           name_suffix=".npz",
                           hash_keywords=["tstart", "tstop"])
def get_time_intervals_for_program(signal_name, id,
                                   useSingleInterval=False,
                                   tstart=None, tstop=None,
                                   timeout=1, protocol="json",
                                   cacheSettings=None, useCache=True):
    """
        Find time intervals with data for a given program id.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        id : str
            Program id, of the form "20160310.002".
        useSingleInterval : bool, optional
            To minimized calls to archivedb, do not request the interval
            list but simply return program from to.
        tstart: float, optional
            Limit time intervals to be later than this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        tstop: float, optional
            Limit time intervals to be before this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        Returns
        -------
        out : nd.array
            2d array of all time intervals where the data samples exist.
    """
    # note that presently we don't use useLastVersion here, bcs. usually
    # this function is already called with a proper url
    time_from, time_to = programs.get_program_from_to(id, timeout=timeout,
                                                      protocol=protocol,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)
    if time_from == 0:
        return [[]]
    if tstart is not None or tstop is not None:
        t1 = programs.get_program_t1(id, timeout=timeout, protocol=protocol,
                                     useCache=useCache,
                                     cacheSettings=cacheSettings)
        if tstart is not None:
            time_from = t1 + tstart * 1000000000
        if tstop is not None:
            time_to = t1 + tstop * 1000000000
    if useSingleInterval:
        logging.getLogger(__name__).debug("Using single interval mode. ")
        return np.array([[time_from, time_to]])
    else:
        return get_time_intervals(signal_name, time_from, time_to,
                                  timeout=timeout, protocol=protocol,
                                  useCache=useCache,
                                  cacheSettings=cacheSettings)
