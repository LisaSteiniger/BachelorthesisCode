# -*- coding: utf-8 -*-
"""
    Set of functions dealing with parameter logs.
"""
import numpy as np
try:
    import configparser  # Python 3
except ImportError:
    import ConfigParser as configparser  # Python 2
import collections
import logging
from . import cache
from . import url
from . import utils
from . import versions
from . import programs


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PARLOG,
                           hash_keywords=["mapToArray"],
                           safe_context=lambda *args, **kwargs: True)
# settings={"useMemCache": True,
# "maxCacheLength": 10})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PARLOG,
                           hash_keywords=["mapToArray"],
                           safe_context=lambda *args, **kwargs: True,
                           name_prefix="parlog",
                           name_suffix=".txt")
def get_parameters_box(signal_name, time_from, time_to,
                       mapToArray=False, timeout=1,
                       protocol="json", cacheSettings=None, useCache=True):
    """
        Get parameter boxes  within a given time interval.

        Parameters
        ----------
        signal_name : string
            The signal adress.
        time_from, time_to: int or string
            Time boundaries of the form 'YYYY-MM-DD HH:MM:SS.%f'
            or nanosecond time stamp.
        mapToArray: boolean, optional
            If True, the server is asked to return arrays instead of mapped
            dictionaries.

        Returns
        -------
        out : dict
            Set of parameters.

        Examples
        --------
        >>> get_parameters_box("ArchiveDB/raw/W7X/CoDaStationDesc...",
                                1438604887140000000, 1438604887240000000)
    """

    baseaddress = url._make_string_signal_name(signal_name)
    baseaddress = baseaddress + "/_signal.%s" % protocol
    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)
    address = (baseaddress + '?from=' + str(int(time_from)) +
               '&upto=' + str(int(time_to)))
    if mapToArray:
        address = address + "&mapToArray=true"
    d = url.read_from_url_and_decode(address, timeout=timeout,
                                     protocol=protocol)
    return d


@versions.versionize_signal
@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.PARLOG,
                           hash_keywords=["mapToArray"],
                           safe_context=lambda *args, **kwargs: False)
# settings={"useMemCache": True,
# "maxCacheLength": 10})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.PARLOG,
                           hash_keywords=["mapToArray"],
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="parlog",
                           name_suffix=".txt")
def get_parameters_box_for_program(signal_name, id,
                                   mapToArray=False,
                                   timeout=1, protocol="json",
                                   useLastVersion=False, version=None,
                                   updateURLVersion=False, returnVersion=False,
                                   returnSignalPath=False,
                                   cacheSettings=None, useCache=True):
    """
        Get the parameter box  within the time interval of a program.

        Parameters
        ----------
        signal_name : string
            The signal address.
        id : str
            Program id, of the form "20160310.002".
        mapToArray: boolean, optional
            If True, the server is asked to return arrays instead of mapped
            dictionaries.
        timeout, protocol - see url.read_from_url_and_decode
        useLastVersion, version, updateURLVersion, returnVersion,
        returnSignalPath - see versions.add_version_as_required
        useCache, cacheSettings - control cache behaviour
        Returns
        -------
        out : dict
            Set of parameters.

        Examples
        --------
        >>> get_parameters_box("ArchiveDB/raw/W7X/CoDaStationDesc...",
                               "20171017.008")
    """
    protocol = "json"  # cbor is not working presently

    time_from, time_to = programs.get_program_from_to(id, timeout=timeout,
                                                      protocol=protocol,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)
    if time_from == 0:
        return {}
    return get_parameters_box(signal_name, time_from, time_to,
                              mapToArray=mapToArray,
                              timeout=timeout, protocol=protocol,
                              useCache=useCache,
                              cacheSettings=cacheSettings)


def write_parameters(signal_name, data_dict, time_interval, protocol="json"):
    """
        Writes parameter log (_PARLOG) into database.

        Parameters
        ----------
        signal_name : string
            The signal address.
        data_dict : dict
            Parametes as a Python dictionary of the form:
            Example:
                    { 'chanDescs': {
                                        '[0]':{
                                            'name' : 'Alpha', 
                                            'active' : 1, 
                                            'physicalQuantity' : { 'type' : 'X'}
                                        },
                                        '[1]' : 
                                        {
                                            'name' : 'Beta', 
                                            'active' : 1, 
                                            'physicalQuantity' : { 'type' : 'X'}
                                        }
                                                 },
                       'powerLevel' : 100 }

        time_interval : 2 values list
            Time intervals when parameters are valid: ns timestamps

        Returns
        -------
        out : boolean
            True if the data is successfully created.

        Examples
        --------
        >>> name = 'Test/raw/W7XAnalysis/webapi-tests/pythonTest1_PARLOG/'
        >>> t = int(round(time.time() * 1000)) * 1000000
        >>> write_parameters('Test',name, para_data, [t, -1])
    """
    url_ = url._make_full_url(signal_name)
    logger = logging.getLogger(__name__)
    logger.info("write parlogs to url " + url_)

    msg = {"label": "parms", "values": [data_dict, ],
           "dimensions": time_interval}
    para_msg = url._afterencoder[protocol](url._encoder[protocol](msg))
    logger.info("send message")  # + para_msg)

    request = url.urllib2.Request(url_, data=para_msg,
                                  headers={"Content-type": "application/%s" % protocol})
    try:
        logger.info("send data")
        response = url.urllib2.urlopen(request)
    except url.urllib2.HTTPError as ex:
        logger.exception("writing parlog failed")
        msg = ex.read()
        raise RuntimeError(msg)
    return (response.code == 201)


def write_parameters_for_program(signal_name, data_dict, shot,
                                 tstart=None, tstop=None, protocol="json",
                                 timeout=1, useCache=True, cacheSettings=None):
    """
        Write PARLOG for a shot (possibly within a time interval in the shot),

        Parameters
        ----------
        signal_name : string
            The signal address.
        data_dict : dict
            Parametes as a Python dictionary of the form:
        shot : str
            Shot number, e.g. 20180920.013
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
        out : boolean
            True if the data is successfully created.
    """
    time_from, time_to = programs.get_program_from_to(
        shot, timeout=timeout, protocol=protocol, useCache=useCache,
        cacheSettings=cacheSettings)
    if time_from == 0:
        raise RuntimeError("Could not find valid time interval for "
                           "program %s. Probably this shot is not "
                           "valid." % shot)
    if tstart is not None or tstop is not None:
        t1 = programs.get_program_t1(shot, timeout=timeout,
                                     protocol=protocol, useCache=useCache,
                                     cacheSettings=cacheSettings)
        if tstart is not None:
            time_from = int(t1 + tstart * 1000000000)
        if tstop is not None:
            time_to = int(t1 + tstop * 1000000000)
    return write_parameters(signal_name, data_dict, [time_from, time_to],
                            protocol=protocol)


def write_parameters_from_ini_file(streamGroup, fileName, time_interval,
                                   add=None, addChannels=None, protocol="json"):
    """
        Read parameters from an ini file and write them to a parameter log (_PARLOG).

        Creates a settings dictionary form an ini file. An ini file looks like:
        [Section 1]
        par 1 = value1
        par 2 = value2
        [Section 2]
        .....
        Sections that must be present are: Info and Channel X, where X are channel
        numbers.  The Info sections must have entries name and serial number,
        they are used to form the full signal name.

        Parameters are written to: streamGroup/name_serialNumber_PARLOG.

        Parameters
        ----------
        streamGroup : string
            The stream address.
        fileName : string
            Name of the input ini file.
        time_interval : 2 values list
            Time intervals when parameters are valid: ns timestamps
        add : dictionary
            An optional dictionary of additional settings to add.
        addChannels: dictionary
            An optional dictionary of additional settings that is applied for
            every channel entry in chanDescs.

        Returns
        -------
        out : boolean
            True if the data is successfully created.

        Examples
        --------
        >>> write_parameters_from_ini_file('Test/raw/W7X/thoms_test',
                                            "ADQ14DC_SPD-04385.ini", [1, -1])
    """
    logger = logging.getLogger(__name__)
    logger.info("read input ini config " + fileName)
    config = configparser.ConfigParser()
    config.read(fileName)

    logger.info("transfer config settings to dictionary")
    settings = {}
    # important! DB expects proper order
    settings["chanDescs"] = collections.OrderedDict()
    channels = []
    for s in config.sections():
        if s[:7] == "Channel":
            n = int(s.split(" ")[-1])
            ref = "[%d]" % n
            settings["chanDescs"][ref] = {}
            settings["chanDescs"][ref]["name"] = "channel_%d" % n
            settings["chanDescs"][ref]["active"] = 1
            for opt in config.options(s):
                settings["chanDescs"][ref][opt] = config.get(s, opt)
            if addChannels is not None:
                for k in addChannels.keys():
                    settings["chanDescs"][ref][k] = addChannels[k]
            continue
        settings[s] = {}
        for opt in config.options(s):
            settings[s][opt] = config.get(s, opt)

    if add is not None:
        logger.info("append explicitly passed settings")
        settings.update(add)
    if streamGroup[-1] == '/':
        streamGroup = streamGroup[:-1]

    logger.info("finished preparing settings dictionary")
    logger.debug(settings)
    name = "%s/%s_%s_PARLOG" % (streamGroup, config.get("Info", "name"),
                                config.get("Info", "serial number"))
    return write_parameters(name, settings, time_interval, protocol=protocol)
