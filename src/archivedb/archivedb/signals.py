# -*- coding: utf-8 -*-
"""
    Set of functions dealing with signals.
"""
import numpy as np
import logging
import functools
from . import cache
from . import url
from . import utils
from . import programs
from . import versions
from . import timing
from . import aliases
from . import parlogs

_dtypes = {"byte": int,  # Python figures out the length
           "short": int,
           "int": int,
           "integer": int,
           "long": int,
           "float": float,
           "double": float,  # Python float is C double
           "boolean": bool,
           "char": str,
           "string": str}

# TODO
# 3. add getting reduced resolution
# 4. consider switching to * keyword separator and dropping py2 support,
# bcs. otherwise there is a cache bug if e.g. some args are passed as
# keywords.


# A decorator to add time correction on the top of the stack, so that
# cached signals are in ns format. This assumes, that shot is the second
# input argument and the time vector is the first in the return list.
def _correct_time_to_plasma_start(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        correct_time = kwargs.get("correctTime", True)
        res = func(*args, **kwargs)
        if not correct_time:
            return res
        id = args[1]
        logging.getLogger(__name__).debug("correct time to plasma start for "
                                          "id " + id)
        timeout = kwargs.get("timeout", 1)
        protocol = kwargs.get("protocol", "json")
        useCache = kwargs.get("useCache", True)
        cacheSettings = kwargs.get("cacheSettings", None)
        # res = list(res)
        t0 = programs.get_program_t0(id, timeout=timeout, protocol=protocol,
                                     useCache=useCache,
                                     cacheSettings=cacheSettings)
        return ((res[0] - t0)*1e-9 - 60.0,) + res[1:]
        # it is the same, but creates a temporary variable
        # t = res[0]
        # t = (t - t0)*1e-9 - 60.0
        # res[0] = t
        # return tuple(res)
    return wrap


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: True,
                           hash_keywords=["enforceDataType"])
# settings={"useMemCache": False,
# "maxCacheLength": 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: True,
                           hash_keywords=["enforceDataType"],
                           name_prefix="data",
                           name_suffix=".npz")
def get_signal(signal, time_from, time_to, enforceDataType=False,
               timeout=1, protocol="json",
               cacheSettings=None, useCache=True):
    """
        Get the signal data within a given time interval.

        It is recommended to call this function with valid interval
        boundaries. In such a way the caching can be used effectively.

        Parameters
        ----------
        signal: str
            The signal address.
        time_from: int or string
            Nanosecond time stampt or string of the form 'YYYY-MM-DD HH:MM:SS.%f'.
        time_to: int or string
            Nanosecond time stampt or string of the form 'YYYY-MM-DD HH:MM:SS.%f'.
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        timeout : int, optional
            URL timeout.
        protocol : str
            Document protocol json or cbor.
        useCache : bool
            Use caching.
        cacheSettings : dict or None
            Caching settings - this is the new interface that will replace
            useCache.

        Returns
        -------
        times, values : two arrays
            Time stamp and value arrays.

        See Also
        --------
        get_latest_time_interval : Return latest valid time stamp interval
        get_time_intervals : Return valid time stamp intervals


        Examples
        --------
        >>> signal = 'ArchiveDB/raw/W7X/CoDaStationDesc...'
        >>> get_signal(signal, 1439391817693999999,1439391817694000000)
    """
    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)

    address = url._make_string_signal_name(signal) + "/_signal.%s" % protocol
    address = (address+'?from=' +
               str(int(time_from)) + '&upto=' + str(int(time_to)))

    d = url.read_from_url_and_decode(address, timeout=timeout,
                                     protocol=protocol)
    x = np.array(d['dimensions'])
    y = np.array(d['values'])
    if enforceDataType:
        logger = logging.getLogger(__name__)
        logger.debug("explicit type conversion requested")
        x = x.astype(np.int64)
        if "datatype" in d and d["datatype"]:
            logger.debug("convert to " + d["datatype"].lower())
            y = y.astype(_dtypes.get(d["datatype"].lower(), np.float64))
        else:
            logger.debug("no, or invalid data type, convert to float")
            y = y.astype(np.float64)
    del d
    return x, y

# we don't cache this one, bcs. it's cached by get_signal and
# hasher is not designed for 2d lists.


def get_signal_multiinterval(signal_name, intervals, enforceDataType=False,
                             useCache=True,
                             timeout=1, protocol="json", cacheSettings=None):
    """
        Get the signal data  for multiple intervals.

        Note: intervals are inverted, to be usable together with get_time_intervals.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        intervals : 2 dimensional nd.array
            A 2d array of time intervals.
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        useCache, timeout, protocol, cacheSettings are the same as for
        get_signal.

        Returns
        -------
        times, values : ndarray,ndarray
            Time stamp and value arrays. All intervals are joined together.
    """
    app_time = []
    app_signal = []
    for interval in intervals[::-1]:
        if not len(interval):  # can be list or array
            continue
        temp_time, temp_signal = get_signal(signal_name, interval[0],
                                            interval[1], useCache=useCache,
                                            timeout=timeout, protocol=protocol,
                                            cacheSettings=cacheSettings,
                                            enforceDataType=enforceDataType)
        app_time.extend(temp_time)
        app_signal.extend(temp_signal)
        del temp_time, temp_signal
    return np.array(app_time), np.array(app_signal)


@aliases.convert_alias_to_signal
@versions.versionize_signal
@_correct_time_to_plasma_start
@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: False,
                           hash_keywords=["enforceDataType",
                                          "tstart", "tstop"])
# settings={"useMemCache": False,
# "maxCacheLength": 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: False,
                           hash_keywords=["enforceDataType",
                                          "tstart", "tstop"],
                           name_prefix="data",
                           name_suffix=".npz")
def get_signal_for_program(signal_name, id, correctTime=True,
                           enforceDataType=False,
                           useSingleInterval=False,
                           tstart=None, tstop=None,
                           timeout=1, protocol="json",
                           useLastVersion=False, version=None,
                           updateURLVersion=False, returnVersion=False,
                           returnSignalPath=False,
                           useCache=True, cacheSettings=None):
    """
        Return data for given program.

        Parameters
        ----------
        signal_name : string
            The signal address.
        id : str
            Program id, of the form "20160310.002".
        correctTime: bool, optional
            If true the time vector is aligned to the plasma start
            and converted to seconds, i.e. (t - t0)*1e-9 - 60.0
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        useSingleInterval : bool, optional
            To minimized calls to archivedb, do not request data for
            each store interval, but instead send a request for the full
            program time. This can potentially reduce cache hits, but
            may be desirable for the speed.
        tstart: float, optional
            Limit time intervals to be later than this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        tstop: float, optional
            Limit time intervals to be before this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        timeout, protocol - see url.read_from_url_and_decode
        useLastVersion, version, updateURLVersion, returnVersion,
        returnSignalPath - see versions.add_version_as_required and
        versions.versionize_signal
        useCache, cacheSettings - control cache behaviour

        Returns
        -------
        times, values : ndarray,ndarray
            Time stamp and value arrays.
    """
    logger = logging.getLogger(__name__)

    intervals = timing.get_time_intervals_for_program(signal_name, id,
                                          useSingleInterval=useSingleInterval,
                                                      tstart=tstart,
                                                      tstop=tstop,
                                                      timeout=timeout,
                                                      protocol=protocol,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)
    if len(intervals) == 1 and len(intervals[0]) == 0:
        logger.debug("no valid time intervals found, return empty result")
        return np.array([]), np.array([])
    t, s = get_signal_multiinterval(signal_name, intervals, useCache=useCache,
                                    timeout=timeout, protocol=protocol,
                                    cacheSettings=cacheSettings,
                                    enforceDataType=enforceDataType)
    return t, s


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: True,
                           hash_keywords=["enforceDataType"])
# settings={"useMemCache": False,
# "maxCacheLength": 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: True,
                           hash_keywords=["enforceDataType"],
                           name_prefix="data",
                           name_suffix=".npz",
                           save_input_parameters=True,
                           validate_cache=cache._is_valid_input_string_in_cache,
                           compose_cache_filename=cache._compose_hash_cache_filename_with_signame)
def get_signal_box(signal_name, time_from, time_to, channels,
                   enforceDataType=False,
                   timeout=1, protocol="json",
                   cacheSettings=None, useCache=True):
    """
        Get signal box (multichannel set) for the given interval.

        Parameters
        ----------
        signal_name : string
            The signal address.
        time_from, time_to: int or string
            Time boundaries of the form 'YYYY-MM-DD HH:MM:SS.%f',
            or nanosecond time stamp.
        channels: list
            The channels for which data is required
            Of the form [3,4]
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        useCache, cacheSettings, timeout, protocol are the same as for get_signal.

        Returns
        -------
        times, values : ndarray,ndarray
            Time stamp and value arrays.

        Examples
        --------
        >>> get_signal_box("ArchiveDB/raw/W7X/CoDaStationDesc....",
                            1407920479822000000, 1407920481823000000,
                            [139,140,141])
    """
    logger = logging.getLogger(__name__)
    # if not channels:
    if channels is None or len(channels) < 1:
        logger.warning("empty channel list, return empty result")
        return np.array([]), np.array([])

    baseaddress = url._make_string_signal_name(signal_name)
    address = baseaddress + "/_signal.%s" % protocol

    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)

    # Note, we download the full set of requested channels and
    # cache and return all of them together.
    # This is different from the previous version, where we
    # kept individual caches per channel and tried to download only missing.
    # The new approach is more uniform for the caching layer and is in any case
    # the most used. The only downside is a possible increase of the cache size.
    address = (address + '?from=' + str(int(time_from)) + '&upto=' +
               str(int(time_to)) + "&channels=" +
               ",".join([str(c) for c in channels]))

    d = url.read_from_url_and_decode(address, timeout=timeout,
                                     protocol=protocol)
    dim = np.array(d['dimensions'])
    val = np.array(d['values'])

    if enforceDataType:
        logger.debug("explicit type conversion requested")
        dim = dim.astype(np.int64)
        if "datatype" in d and d["datatype"]:
            logger.debug("convert to " + d["datatype"].lower())
            val = val.astype(_dtypes.get(d["datatype"].lower(), np.float64))
        else:
            logger.debug("no, or invalid data type, convert to float")
            val = val.astype(np.float64)

    del d
    return dim, val

# Don't cache this one, this is already cached in the single interval version.


def get_signal_box_multiinterval(signal_name, intervals, channels,
                                 enforceDataType=False,
                                 useCache=True, timeout=1, protocol="json",
                                 cacheSettings=None):
    """
        Read signal box for multiple time intervals.

        Note: intervals are inverted, to be usable together with get_time_intervals.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        intervals : 2 dimensional nd.array
            A 2d array of time intervals.
        channels: list
            Channels for which data is required, e.g. [3,4]
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        timeout, protocol, useCache, cacheSettings as for get_signal

        Returns
        -------
        times, values : ndarray,ndarray
            Time stamp and value arrays. All intervals are joined together.
    """
    baseaddress = url._make_string_signal_name(signal_name)

    app_time = []
    app_signal = []

    for i in range(len(channels)):
        app_signal.append([])

    for interval in intervals[::-1]:
        if not len(interval):
            continue
        temp_time, temp_signal = get_signal_box(baseaddress, interval[0],
                                                interval[1], channels,
                                                useCache=useCache,
                                                timeout=timeout,
                                                protocol=protocol,
                                                cacheSettings=cacheSettings,
                                                enforceDataType=enforceDataType)
        app_time.extend(temp_time)
        for i in range(len(channels)):
            # app_signal[i].extend(temp_signal[i].tolist()) # ?
            app_signal[i].extend(temp_signal[i])
        del temp_time, temp_signal
    return np.array(app_time), np.array(app_signal)


@versions.versionize_signal
@_correct_time_to_plasma_start
@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: False,
                           hash_keywords=["enforceDataType",
                                          "tstart", "tstop"])
# settings={"useMemCache": False,
# "maxCacheLength": 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.SIGNAL,
                           safe_context=lambda *args, **kwargs: False,
                           hash_keywords=["enforceDataType",
                                          "tstart", "tstop"],
                           name_prefix="data",
                           name_suffix=".npz",
                           save_input_parameters=True,
                           validate_cache=cache._is_valid_input_string_in_cache,
                           compose_cache_filename=cache._compose_hash_cache_filename_with_signame)
def get_signal_box_for_program(signal_name, id, channels, correctTime=True,
                               enforceDataType=False,
                               useSingleInterval=False,
                               tstart=None, tstop=None,
                               timeout=1, protocol="json",
                               useLastVersion=False, version=None,
                               updateURLVersion=False, returnVersion=False,
                               returnSignalPath=False,
                               cacheSettings=None, useCache=True):
    """
        Return signal box data for a given program.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        id : str
            Program id, of the form "20160310.002".
        channels: list
            Channels for which data is required, e.g. the form [3,4].
        correctTime: bool, optional
            If true the time vector is aligned to the plasma start,
            i.e. (t - t0)*1e-9 - 60.0
        enforceDataType : bool, optional
            Use received data type information to force data type of the output
            array. If no datatype is received or it is empty, float is used.
        useSingleInterval : bool, optional
            To minimized calls to archivedb, do not request data for
            each store interval, but instead send a request for the full
            program time. This can potentially reduce cache hits, but
            may be desirable for the speed.
        tstart: float, optional
            Limit time intervals to be later than this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        tstop: float, optional
            Limit time intervals to be before this time relative to the
            T1 trigger (i.e. to the plasma start). This time should be in
            seconds.
        timeout, protocol - see url.read_from_url_and_decode
        useLastVersion, version, updateURLVersion, returnVersion,
        returnSignalPath - see versions.add_version_as_required
        useCache, cacheSettings - control cache behaviour

        Returns
        -------
        times, values : ndarray,ndarray
            Time stamp and value arrays. All intervals are joined together.
    """
    logger = logging.getLogger(__name__)
    # The real difficulty is to get the available intervals.
    # One needs a full channel name to ask for intervals.
    # Try to get one.
    logger.debug("Find name of the first channel to use it for time intervals")
    baseaddress = url._make_string_signal_name(signal_name)
    address = baseaddress + "/{:d}".format(channels[0])
    time_from, time_to = programs.get_program_from_to(id, timeout=timeout,
                                                      protocol=protocol,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)

    d = url.read_from_url_and_decode((address + "?filterstart=" + str(time_from) +
                                      "&filterstop=" + str(time_to)),
                                     timeout=timeout)
    sname = d["_links"]["children"][0]["href"].split("?")[0]
    logger.debug("first channel name is " + sname)

    logger.debug("find time intervals")
    intervals = timing.get_time_intervals_for_program(sname, id,
                                          useSingleInterval=useSingleInterval,
                                                      tstart=tstart,
                                                      tstop=tstop,
                                                      timeout=timeout,
                                                      protocol=protocol,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)
    if len(intervals) == 1 and len(intervals[0]) == 0:
        logger.debug("no intervals, return empty data")
        return np.array([]), np.array([])

    logger.debug("get data for all channels for all intervals")
    t, s = get_signal_box_multiinterval(signal_name, intervals,
                                        channels, useCache=useCache,
                                        timeout=timeout, protocol=protocol,
                                        cacheSettings=cacheSettings,
                                        enforceDataType=enforceDataType)
    return t, s


def write_signal(signal_name, timestamp_list, data_list, protocol="json",
                 data_format=None, datatype=None):
    """
        Writes signal to database.

        Supplied data array is not carefully checked at the client side to
        maintain the flexibility, i.e. to cover all possible use cases. One can
        use this url:
            http://archive-webapi.ipp-hgw.mpg.de/validation.html
        to verify data format. Most common cases are:
            - 1d array of the same length as the number of timestamps
            - 2d array [channel, time], where the first dimension should be of
              the same size as the number of channels in the stream and the
              second dimension should correspond to time stamps
            - 3d array [time, channel, profile point], here the first dimension
              should be of the same length as the time vector, the second
              dimension should cover all channels of the stream and the last
              dimension normally corresponds to a profile point. In this case
              data_format should be set to  "multichannel_profile" and datatype
              to "float".

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        timestamp_list : list
            The list of time in nanoseconds, of the form:
            [1435601915999999999,1435601915999999998,1435601915999999992]
        data_list : list
            The signal data, of the form: [1, 2, 3]
        data_format, datatype - If present are added to submitted JSON, e.g.
        for writing profile data.

        Returns
        -------
        out : boolean
            True if the data is successfully created.

        Examples
        --------
        >>> my_timestamp = int(round(time.time() * 1000)) * 1000000
            timestamps=[my_timestamp, my_timestamp+1, my_timestamp+2,
                        my_timestamp+3,my_timestamp+4]
            write_signal('Test/raw/W7XAnalysis/webapi...', timestamps,
                         [11,2,3,4,15])
    """
    logger = logging.getLogger(__name__)
    logger.info("writing signal to data base")

    if not isinstance(timestamp_list, list):
        timestamp_list = timestamp_list.tolist()
    if not isinstance(data_list, list):
        data_list = data_list.tolist()
    url_ = url._make_full_url(signal_name)

    shape = np.array(data_list).shape
    if len(shape) == 3:
        logger.debug("3d data box is supplied, check the first dimension "
                     "and add multichannel_profile")
        if shape[0] != len(timestamp_list):
            msg = ("for 3d box data the first dimension should be of "
                   "the length as the time vector, time length is %d,"
                   "data shape is %s" % (len(timestamp_list), str(shape)))
            logger.error(msg)
            raise RuntimeError(msg)
        if data_format is None:
            logger.debug("set data_format to multichannel_profile")
            data_format = "multichannel_profile"
        if datatype is None:
            logger.debug("set datatype to float")
            datatype = "float"

    msg = {'values': data_list,
           'dimensions': timestamp_list}
    if data_format is not None:
        msg["data_format"] = data_format
    if datatype is not None:
        msg["datatype"] = datatype
    msg = url._afterencoder[protocol](url._encoder[protocol](msg))
    request = url.urllib2.Request(url_, data=msg,
                                  headers={"Content-type":
                                           "application/%s" % protocol})
    try:
        logger.info("send data")
        response = url.urllib2.urlopen(request)
    except url.urllib2.HTTPError as ex:
        logger.exception("Failed to write signal")
        msg = ex.read()
        raise RuntimeError(msg)

    return (response.code == 201)


def write_versioned_signal_with_parlog(stream, shot, times, data, num_channels,
                                       relative_time=True, parlog=None,
                                       channel_names=None, user_info=None,
                                       data_format=None, datatype=None,
                                       target_version=None,
                                       protocol="json", timeout=1):
    """
        Writes signal (signal box) and parlog simultaneously to the database,
        updating versions if necessary.

        This is a convenience function that should greatly simplify the
        complexity of writing parlogs, signals and updating versions.
        Internally this get_next_sync_version_for_shot to find the next version
        to write and than uses write_signal and write_parlog with versioned
        streams.

        Supplied data array is not carefully checked at the client side to
        maintain the flexibility, i.e. to cover all possible use cases. One can
        use this url:
            http://archive-webapi.ipp-hgw.mpg.de/validation.html
        to verify data format. Most common cases are:
            - 1d array of the same length as the number of timestamps
            - 2d array [channel, time], where the first dimension should be of
              the same size as the number of channels in the stream and the
              second dimension should correspond to time stamps
            - 3d array [time, channel, profile point], here the first dimension
              should be of the same length as the time vector, the second
              dimension should cover all channels of the stream and the last
              dimension normally corresponds to a profile point. In this case
              data_format should be set to  "multichannel_profile" and datatype
              to "float".

        Parameters
        ----------
        stream : string
            Stream to write, can contain DATASTREAM or PARLOG suffix, which
            will be removed as necessary.
        shot: string
            W7-X shot number for which to upload the data
        times : list, array
            List of time points for which data are to be written.
        data: list, array
            Data to be written. Note that if datastream contains multiple
            channels, all data should be written at once.
        num_channels: int
            Number of channels in the data set. This is used to form the
            chanDescs section in the parlog, if not present explicitly. Note
            that if chanDescs is present in the supplied parlog, this argument
            is not used and can be set to an arbitrary value.
        relative_time: boolean
            If True (default), the time vector is considered to be is seconds,
            relative to shot T1 and is converted to ns before uploading. If
            False, not conversion is done.
        parlog: dict
            Parlog to be written. This may or may not contain chanDescs. In the
            latter case the chanDescs scection is created using number of
            channels and automatic channel naming or names from channel_names.
        channel_names: list
            Name of channels to use for the chanDescs section. If the length of
            this list is smaller than num_channels, automatic names are
            used for the missing channels.
        user_info: dict
            Additional information to add to the parlog.
        target_version: str, optional
            If None, the next suitable version will be used, i.e. will be
            created if necessary. If a target version is of the form V2, the
            function will attempt to write to this version, if it is available.
            If the desired version is occupied, an error will be raised.
        data_format, datatype: str
            Can be used to provide details of the data and are sent to the
            archive.
        timeout:
            Timeout is not properly supported yet.

        Returns
        -------
        out : str
            Version number of the form V1 if the data was successfully
            uploaded.
    """
    logger = logging.getLogger(__name__)
    logger.info("write versioned signal with parlog to stream %s, for "
                 "shot %s" % (stream, shot))

    base_stream = url._get_base_stream_name(stream)
    parlog_stream = base_stream + "_PARLOG"
    data_stream = base_stream + "_DATASTREAM"
    logger.debug("base stream name " + base_stream)
    logger.info("parlog stream name " + parlog_stream)
    logger.info("datastream stream name " + data_stream)

    if relative_time:
        logger.info("relative time suppled, convert to absolute time stamps")
        t1 = programs.get_program_t1(shot)
        logger.debug("shot T1 is %d" %t1)
        times = [t1 + int(x * 1000000000) for x in times]

    logger.debug("form parlog")
    if parlog is None:
        parlog = {}
    if "chanDescs" not in parlog:
        logger.debug("form chanDescs section")
        parlog["chanDescs"] = {}
        if channel_names is None:
            channel_names = ["channel %d" % (x+1) for x in range(num_channels)]
        if len(channel_names) < num_channels:
            channel_names.extend(
                [
                    "channel %d" % x for x in
                     range(len(channel_names)+1, num_channels+1)
                 ]
            )
        logger.debug("use channel names %s" % str(channel_names))
        for i, name in enumerate(channel_names):
            parlog["chanDescs"]["[%d]" % i] = {
                "name" : name,
                "active": 1
            }
    if user_info is not None:
        parlog["user_info"] = user_info
    logger.debug("parlog for upload %s" % str(parlog))

    logger.info("find next synced version for writing")
    if target_version is None:
        logger.debug("no target version requested, find next available")
        version = versions.get_next_sync_version_for_shot(base_stream, shot,
                                                          timeout=timeout)
    else:
        logger.debug("target version %s is requested, try to initialize"
                     " it" % str(target_version))
        version = versions.create_target_sync_version_for_shot(
            base_stream, shot, target_version, timeout=timeout)
    logger.debug("version for writing is %s" % version)
    parlog_stream = parlog_stream + "/" + version
    data_stream = data_stream + "/" + version

    logger.info("write parlog %s" % parlog_stream)
    res1 = parlogs.write_parameters_for_program(parlog_stream, parlog, shot,
                                                protocol=protocol,
                                                timeout=timeout)
    if not res1:
        logger.error("failed to write parlog %s to stream %s for shot %s, "
                     "stop" % (str(parlog), parlog_stream, shot))
        return None

    logger.info("write data %s" % data_stream)
    res2 = write_signal(data_stream, times, data, protocol=protocol,
                        data_format=data_format, datatype=datatype)
    if not res2:
        logger.error("failed to write data to stream %s for shot %s, "
                     "stop" % (parlog_stream, shot))
        return None
    return version
