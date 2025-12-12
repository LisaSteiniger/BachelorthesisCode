# -*- coding: utf-8 -*-
"""
    Functions to request shot information.
"""
import time
import re
import logging
from . import utils
from . import url
from . import cache


# TODO:
# 1. change default time out to a longer value
# 2. add option of getting intervals for a time window inside shot
# 3. add parsing specs of different type to unify shot ids and time windows

# A small helper to decide during run time if caching is safe.
# If the requested time window is before midnight of today,
# consider caching safe


_shot_validator = re.compile("^[0-9]{8}\.[0-9]{3}$")
_ts_range_validator = re.compile("^([0-9]{19})_([0-9]{19})$")


def _is_cache_safe(*args, **kwargs):
    midnight = (int(time.time() // 86400)) * 86400
    for arg in args:
        t = utils.to_timestamp(arg)/1e9
        if t > midnight:
            return False
    return True

# The same but with shot number as input.


def _is_cache_safe_for_id(*args, **kwargs):
    id = args[0]
    t = id[:4] + "-" + id[4:6] + "-" + id[6:8] + " 12:00:00"
    return _is_cache_safe(t)


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 100})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           name_prefix="programs/programlist",
                           name_suffix=".txt")
def get_program_list(t0, t1, timeout=5, protocol="json",
                     cacheSettings=None, useCache=True):
    """Get list of performed experimental programs.

       This function returns a list of dictionaries for programs
       performed between start and stop times. The times are to be
       specified as '2016-03-10 12:00:00', or as ns timestamps.
    Parameters
    ----------
    t0 : str
        Start time window.
    t1 : str
        Stop time window

    Returns
    -------
    out: list of dict
        Program list.
    """
    protocol = "json"  # cbor is not working presently
    t0 = utils.to_timestamp(t0)
    t1 = utils.to_timestamp(t1)
    address = url._db_url + "programs.%s?from=%d&upto=%d" % (protocol, t0, t1)
    return url.read_from_url_and_decode(address, timeout=timeout,
                                        protocol=protocol)["programs"]

# don't cache it, the data are cached by the above already


def get_program_list_for_day(day, timeout=5, protocol="json",
                             useCache=True, cacheSettings=None):
    """Get list of performed experimental programs for experimental day.

       This function returns a list of dictionaries for programs
       performed on a given day. The date should be
       specified as '2016-03-10'.
    Parameters
    ----------
    day : str
        Day of interest.

    Returns
    -------
    out: list of dict
        Program list.
    """
    day = day.strip()
    return get_program_list(day + " 00:00:01", day + " 23:59:59",
                            timeout=timeout, protocol=protocol,
                            useCache=useCache, cacheSettings=cacheSettings)


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           name_prefix="programs/programid",
                           name_suffix=".txt")
def get_program_id(t0, dt=2, timeout=5, protocol="json",
                   cacheSettings=None, useCache=True):
    """
        Return program id (aka shot number) from ArchiveDB.

        Parameters
        ----------
        t0 : int or str
            The start time or the first t0 trigger of the run, or the experiment start.
        dt : int
            Time window in minutes. This is used to form a time window
            for searching.

        Returns
        -------
        out: str
            Program id as string. This has the form '20160310.002'
    """

    t1 = t0 + dt*60*1000000000
    progs = get_program_list(t0, t1, timeout=timeout, protocol=protocol,
                             useCache=useCache, cacheSettings=cacheSettings)
    if not(len(progs)):  # empty
        return None
    ind = 0
    for i in range(1, len(progs)):
        if abs(progs[i]["trigger"]["0"][0] - t0) < abs(progs[ind]["trigger"]["0"][0] - t0):
            ind = i
    return progs[ind]["id"]


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe,
                           name_prefix="programs/mdsplusid",
                           name_suffix=".txt")
def get_mdsplus_shot(t0, timeout=1,
                     cacheSettings=None, useCache=True):
    """
        Return MDSPlus shot number closest to the trigger t0.

        This uses webservice query, which may fail.

        Parameters
        ----------
        t0 : int or str
            The first t0 trigger of the run, or the experiment start.
            This can be as nanosecond time stamp or time '2016-03-10 10:00:02.12132'.

        Returns
        -------
        out: str
    """
    t0 = utils.to_timestamp(t0)
    address = url._mdsplus_url + "%d" % t0
    return url.read_from_url_and_decode(address, timeout=timeout)["shotnum"]


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe_for_id,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe_for_id,
                           name_prefix="programs/programT0",
                           name_suffix=".txt")
def get_program_t0(id, timeout=5, protocol="json",
                   cacheSettings=None, useCache=True):
    """
        Return program start as nanosecond time stamp.

        This function returns the first t0 of the program.
        The function takes  program id (aka shot number) in form
        '20160308.003'.

        Parameters
        ----------
        id : str

        Returns
        -------
        out: int
    """
    day = id[:4]+"-"+id[4:6]+"-"+id[6:8]
    progs = get_program_list_for_day(day, timeout=timeout, protocol=protocol,
                                     useCache=useCache,
                                     cacheSettings=cacheSettings)
    for prog in progs:
        if prog["id"] == id:
            t0 = prog["trigger"]["0"][0]
            return t0
    return 0


def get_program_t1(*args, **kwargs):
    """
        Return program T1 trigger, i.e. T0 + 60 s.

        All args and kwargs are passed to get_program_t0.
    """
    return get_program_t0(*args, **kwargs) + 60000000000


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe_for_id,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PROGLIST,
                           safe_context=_is_cache_safe_for_id,
                           name_prefix="programs/programfromto",
                           name_suffix=".txt")
def get_program_from_to(id, timeout=5, protocol="json",
                        cacheSettings=None, useCache=True):
    """
        Return program from and upto time stamps.

        The function takes  program id (aka shot number) in form
        '20160308.03'.

        Parameters
        ----------
        id : str

        Returns
        -------
        from, upto: int, int
    """
    day = id[:4]+"-"+id[4:6]+"-"+id[6:8]
    progs = get_program_list_for_day(day, timeout=timeout, protocol=protocol,
                                     useCache=useCache,
                                     cacheSettings=cacheSettings)
    for prog in progs:
        if prog["id"] == id:
            return prog["from"], prog["upto"]
    return 0, 1

def parse_time_spec(spec, returnPlasmaStart=True, **kwargs):
    """
        Parse time specification an return start/end times.

        Parameters
        ----------
        spec : str
            Program/time specification: either program id as 20181016.037,
            or timestamps for the beginning and end as 1323...._12323...
        returnPlasmaStart : bool
            If True, returns T1 trigger in addition to from and to timestamps
            for programs and repeats start time for timestamps.
        kwargs are passed to archivedb calls to get_program_from_to and
        get_program_t0

        Returns
        -------
        tfrom, tto : ints
        or
        tfrom, tto, T1 : ints
    """
    if _shot_validator.match(spec) is not None:
        tFrom, tTo = get_program_from_to(spec, **kwargs)
        if returnPlasmaStart:
            tStart = get_program_t1(spec, **kwargs)
            return tFrom, tTo, tStart
        return tFrom, tTo

    match = _ts_range_validator.match(spec)
    if match is not None:
        # looks like a nano timestamp range
        tFrom = int(match.group(1))
        tTo = int(match.group(2))
        if returnPlasmaStart:
            return tFrom, tTo, tFrom
        return tFrom, tTo

    raise RuntimeError("Time specification '%s' doesn't look like a "
                       "program 20YYMMDD.SSS or a from/to combination:"
                       " xxxxx_yyyyy")
