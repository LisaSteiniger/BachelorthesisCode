# -*- coding: utf-8 -*-
"""
    Functions to deal with data versions.
"""
import numpy as np
import datetime
import re
import functools
import logging
from . import utils
from . import url
from . import programs
from . import cache
from . import version

# TODO
# 1. add version creation
# 2. add returning version if it is already present in the signal
# 3. add handling of signals with scaled/unscaled

# Helper function to determine is version information cache is safe.
# Consider that cache is safe if a fixed version is supplied.
# For a given version the information does not change. If the
# call fails, an exception will be raised and no cache will be stored
# in any case. If information is requested for all versions, such a call
# is not safe, because a new version might have been added.


def _is_version_info_safe(*args, **kwargs):
    return args[1] is not None


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.VERSIONS,
                           safe_context=_is_version_info_safe,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.VERSIONS,
                           safe_context=_is_version_info_safe,
                           name_prefix="versioninfo",
                           name_suffix=".txt")
def get_version_info(stream_name, version, timeout=1, protocol="json",
                     cacheSettings=None, useCache=True):
    """
        Get version information.

        Parameters
        ----------
        stream_name: str
            Name of a data or parlog stream.
        version: str or int
            Version number to get info for. Either a string like 'V1' or int as 1,2.
            If None, info for all versions is returned

        Returns
        -------
        out: dict
            Version information.
    """
    protocol = "json"  # cbor is not working presently
    address = url._make_string_signal_name(stream_name)
    if address[-1] == "/":
        address = address[:-1]
    if isinstance(version, ("".__class__, u"".__class__)):
        address = address + "/"+version+"/_versions.%s" % protocol
    elif isinstance(version, int) or isinstance(version, np.int64):
        address = address + "/V"+str(version)+"/_versions.%s" % protocol
    elif version is None:
        address = address + "/_versions.%s" % protocol
    return url.read_from_url_and_decode(address, timeout=timeout,
                                        protocol=protocol)


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.VERSIONS,
                           safe_context=lambda *args, **kwargs: False,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.VERSIONS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="lastversion",
                           name_suffix=".txt")
def get_last_version(stream_name, time_from, time_to, timeout=1,
                     protocol="json", cacheSettings=None, useCache=True):
    """
        Find the latest available version in the given time interval.

        Parameters
        ----------
        stream_name: str
            Name of data of parlog stream.
        time_from: int or string
            The time from which to search.
            Of the form 'YYYY-MM-DD HH:MM:SS.%f', or nanosecond time stamp.
        time_to: int or string
            The time until which to search.
            Of the form 'YYYY-MM-DD HH:MM:SS.%f', or nanosecond time stamp.
            If None is supplied, current time is used.

        Returns
        -------
        out: str
            Version number like V1 or None.
    """
    logger = logging.getLogger(__name__)
    logger.debug("find last version for stream " + stream_name)
    protocol = "json"  # cbor is not working presently
    address = url._make_string_signal_name(stream_name)
    if address[-1] == "/":
        address = address[:-1]
    time_from = utils.to_timestamp(time_from)
    if time_to is None:
        logger.debug("top limit is None, use current time")
        time_to = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    time_to = utils.to_timestamp(time_to)
    address = (address + "/_versions.%s?from=" % protocol +
               str(time_from) + "&upto=" + str(time_to))
    d = url.read_from_url_and_decode(address, timeout,
                                     protocol=protocol)["versionInfo"]
    if len(d) and d[0].get("creation_tag", 0) < 4102444800000000000:
        logger.debug("valid return result from http")
        return "V"+str(d[0]["number"])
    else:
        logger.debug("No version returned: likely unversioned stream. "
                     "Return None.")
        return None


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.VERSIONS,
                           safe_context=lambda *args, **kwargs: False,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 1000})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.VERSIONS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="lastversion",
                           name_suffix=".txt")
def get_last_version_for_program(stream_name, id,
                                 timeout=1, protocol="json",
                                 cacheSettings=None, useCache=True):
    """
        Find the latest available version for a program.

        Parameters
        ----------
        stream_name: str
            Name of a data or parlog stream.
        id : str
            Program id, of the form "20160310.002".

        Returns
        -------
        out: str
            Version number like V1.
    """
    time_from, time_to = programs.get_program_from_to(id, protocol=protocol,
                                                      timeout=timeout,
                                                      useCache=useCache,
                                                      cacheSettings=cacheSettings)
    if time_from == 0:
        return None
    return get_last_version(stream_name, time_from, time_to, timeout=timeout,
                            protocol=protocol, useCache=useCache,
                            cacheSettings=cacheSettings)


# helper to check if the input string is a version tag
_version_re = re.compile("^V[0-9]+$")


def add_version_as_required(signal, id, version=None,
                            useLastVersion=False,
                            updateURLVersion=False,
                            timeout=1, protocol="json",
                            useCache=True, cacheSettings=None):
    """Appends/changes version to/in the input signal name.

    Parameters
    ----------
    signal : str or list
        Signal name.
    version : str or int, optional
        Required version number. Supported forms: 1, "v1", "V1".
    useLastVersion : bool, optional
        Find last available version and append it.
    updateURLVersion : bool, optional
        If True change version present in the signal name. If False don't
        change anything,  but write a warning.

    Returns
    -------
    out, version : str
        Modified signal name and version (can be None)
    """
    logger = logging.getLogger(__name__)
    logger.debug("handle version information for stream " + signal)

    # if useLastVersion and explicit version present -> raise error
    if version is not None and useLastVersion:
        logger.error("Both explicit version and useLastVersion are present. "
                     "This combination is ambiguous, raising an error.")
        raise RuntimeError("Conflicting input to versionizer. Both "
                           "version and useLastVersion are present.")
    # nothing to be done
    if version is None and not useLastVersion:
        logger.debug("No version and no request for last version, return.")
        return signal, None

    # decompose signal, find its type and find if
    # version information is already included
    logger.debug("find signal type and check if version tag is present")
    version_ind = None  # will be an index to the version tag
    has_version = False
    url_ = url._get_only_signal_name(url._make_string_signal_name(signal))
    path = url_.strip("/").split("/")
    if path[1] == "views":  # alias
        logger.warning(
            "alias supplied, return without modifications: " + signal)
        return signal, None
    if len(path) == 5 and _version_re.match(path[-1].upper()) is None:
        # stream without version
        logger.debug("stream without version")
        path = path + ["V*"]
        version_ind = -1
    elif len(path) == 6 and _version_re.match(path[-1].upper()) is not None:
        # stream with version at the end
        logger.debug("stream with version " + path[-1])
        version_ind = -1
        has_version = True
    elif len(path) == 7 and _version_re.match(path[-3].upper()) is None:
        # channel name without version
        logger.debug("channel name without a version")
        path = path[:-2] + ["V*"] + path[-2:]
        version_ind = -3
    elif len(path) == 8 and _version_re.match(path[-3].upper()) is not None:
        # channel name with version
        logger.debug("channel name with version " + path[-3])
        version_ind = -3
        has_version = True
    else:
        logger.warning("unclear signal, return it and cross fingers for you " +
                       signal)
        return signal, None

    if has_version and not updateURLVersion:
        logger.warning("version present in the input signal, return "
                       " without modifications: " + signal)
        return signal, path[version_ind].upper()

    if useLastVersion:
        logger.debug("request last version information")
        version = get_last_version_for_program(signal, id,
                                               timeout=timeout,
                                               protocol=protocol,
                                               useCache=useCache,
                                               cacheSettings=cacheSettings)
    if version is None:
        logger.debug("no version available, probably unversioned stream")
        return signal, None

    version = str(version).upper()
    if version[0] != "V":
        logger.debug("no V sign, prepend it")
        version = "V" + version
    if _version_re.match(version) is None:
        logger.error("version " + version + "does not "
                     "match the required pattern V[0-9]+, raise an "
                     "error.")
        raise RuntimeError("wrong version format " + version)
    path[version_ind] = version.upper()
    return "/".join(path), version


def versionize_signal(func):
    """
        Decorator to apply add_version_as_required.

        This assumes that signal is the first argument and
        program id is the second one.

        Parameters
        ----------
        returnVersion : boolean
            If True, the used version is appended to the result. None is
            used if no version is found.
        returnSignalPath : boolean
            If True, return the full path appended to the result.
            This is especially useful  after alias resolution. Note,
            this flag takes precedence over returnVersion, bcs. it
            provides more information.
    """
    @functools.wraps(func)
    def inner_function(*args, **kwargs):
        signal = args[0]
        id = args[1]
        version = kwargs.get("version", None)
        useLastVersion = kwargs.get("useLastVersion", False)
        updateURLVersion = kwargs.get("updateURLVersion", False)
        timeout = kwargs.get("timeout", 1)
        protocol = kwargs.get("protocol", "json")
        useCache = kwargs.get("useCache", True)
        cacheSettings = kwargs.get("cacheSettings", None)
        returnVersion = kwargs.get("returnVersion", False)
        returnSignalPath = kwargs.get("returnSignalPath", False)
        signal, version = add_version_as_required(signal, id,
                                                  version=version,
                                                  useLastVersion=useLastVersion,
                                                  updateURLVersion=updateURLVersion,
                                                  timeout=timeout, protocol=protocol,
                                                  useCache=useCache,
                                                  cacheSettings=cacheSettings)
        args = list(args)
        args[0] = signal
        for k in ["version", "useLastVersion", "updateURLVersion",
                  "returnVersion", "returnSignalPath"]:
            kwargs[k] = None # prevent version handling down the stack
        if returnSignalPath:
            return func(*args, **kwargs), args[0]
        elif returnVersion:
            return func(*args, **kwargs), version
        else:
            return func(*args, **kwargs)
    return inner_function


def create_new_version(stream_name, reason=None, producer=None,
                       code_release=None, analysis_environment=None,
                       timeout=1):
    """
        Create new version for the given stream, where stream can be parlog or
        datastream.

        Parameters
        ----------
        stream_name: str
            Signal/parlog address.
        reason, producer, code_release, analysis_environment are strings that
        are sent as version information to the server. If None or an empty
        string is supplied, some fixed values are used to avoid server side
        failures.

        Returns
        -------
        our: str
            Full versionized stream with the created version.
    """
    url_ = url._make_full_url(stream_name)
    logger = logging.getLogger(__name__)
    logger.info("create new version for url %s", url_)
    if url_[-1] == "/":
        url_ = url_[:-1]
    protocol = "json"
    url_ = url_ + "/_versions.%s" % protocol
    if not reason:
        reason = "new version"
    if not producer:
        producer = "python-archivedb"
    if not code_release:
        code_release = version.__version__
    if not analysis_environment:
        analysis_environment = "python"
    msg = {"versionInfo": [{"reason": reason, "producer": producer,
                            "code_release": code_release,
                            "analysis_environment": analysis_environment}]}
    msg = url._afterencoder[protocol](url._encoder[protocol](msg))
    request = url.urllib2.Request(url_, data=msg,
                                  headers={"Content-type":
                                           "application/%s" % protocol})
    try:
        logger.debug("send message: %s", msg)
        response = url.urllib2.urlopen(request, timeout=timeout)
        return url._decoder[protocol](url._predecoder[protocol](
            response))["message"]
    except url.urllib2.HTTPError as ex:
        logger.exception("failed to create new version")
        msg = ex.read().decode()
        raise RuntimeError("failed to create new version: " + msg)



def get_next_version_for_shot(stream_name, shot, timeout=1):
    """
        Returns next version available for writing data for a shot.

        This function checks if there is an empty version that can be used for
        writing data into stream (data stream or parlog), and if not, it
        creates a new version for writing. It should be safe to use the
        returned version for data upload, unless someone else is faster to fill
        the version. This race condition can not be prevented on the client
        side, but requires a reiteration of the procedure.

        Note that this function will not use cache.

        Warnings. Aliases are not handled properly yet. It is unclear what
        happens with non-versioned streams (check). It is likely that urls with
        channel specification will fail, so use root streams instead.

        Parameters
        ----------
        stream_name: str
            Name of data of parlog stream.
        shot : str
            Program id in W7-X format.

        Returns
        -------
        out: str
            Version of the form V1.
    """
    logger = logging.getLogger(__name__)
    logger.debug("find next version for writing for stream " + stream_name)
    protocol = "json"  # safe option, payload is minimal anyway

    # list all versions for the stream
    # not that one can do it with less data transmitted, but it's just more
    # convenient
    versions = get_version_info(
        stream_name, None, timeout=timeout, protocol=protocol,
        useCache=False
    )["versionInfo"]
    max_version = 0
    if versions:
        max_version = max([x["number"] for x in versions])
    logger.debug("maximal created version for this stream is %d" % max_version)

    # find last version already written for the shot
    last_version = get_last_version_for_program(
        stream_name, shot, timeout=timeout, protocol=protocol, useCache=False
    )
    if last_version is None:
        last_version = 0
    else:
        last_version = int(last_version[1:])  # the first symbol should be V
    logger.debug("maximal used version for the shot %s is %d" % (shot,
                                                                 last_version))

    if (max_version == 0) or (last_version == max_version):
        logger.debug("No versions are available or all versions are already "
                     "in use for this shot. Create a new version.")
        # this is a full spec url
        new_version = create_new_version(stream_name,
                                         reason="next version for writing",
                                         timeout=timeout)
        try:
            next_version = int(new_version.split("/")[-1][1:])
        except Exception:
            raise RuntimeError("Failed to extract new version for writing "
                               "after creating a new version. "
                               "The created versionized stream is %s. "
                               "The version is expected to be at the end. "
                               "Please submit a but report." % next_version)
    else:
        logger.debug("Empty versions are available for this shot, use the one "
                     "next to the already used one.")
        next_version = last_version + 1

    logger.debug("Next version is %d" % next_version)

    return "V%d" % next_version


def get_next_sync_version_for_shot(stream_name, shot, timeout=1):
    """
        Returns next version available for writing data for a shot that is
        suitable both for PARLOG and DATASTREAM. That is the version is
        synchronized between PARLOG and DATASTREAM.

        This function relies on get_next_version_for_shot and adds version to
        PARLOG or DATSTREAM if this is behind.

        Parameters
        ----------
        stream_name: str
            Name of data of parlog stream.
        shot : str
            Program id in W7-X format.

        Returns
        -------
        out: str
            Version of the form V1.
    """
    logger = logging.getLogger(__name__)
    logger.debug("find next synced version for writing for "
                 "stream " + stream_name)
    protocol = "json"  # safe option, payload is minimal anyway

    base_stream = url._get_base_stream_name(stream_name)
    parlog_stream = base_stream + "_PARLOG"
    data_stream = base_stream + "_DATASTREAM"
    logger.debug("base stream name " + base_stream)
    logger.debug("parlog stream name " + parlog_stream)
    logger.debug("datastream stream name " + data_stream)

    vparlog = get_next_version_for_shot(parlog_stream, shot, timeout=timeout)
    vdata = get_next_version_for_shot(data_stream, shot, timeout=timeout)
    logger.debug("next parlog version is %s, next datastream "
                 "version is %s" % (vparlog, vdata))
    # next version should be at least 1 at this point, therefore there is no
    # need to check for version None

    if vdata == vparlog:
        logger.debug("versions are synced, return")
        return vdata

    delta = int(vdata[1:]) - int(vparlog[1:])
    if delta > 0:
        logger.debug("next data version is above the parlog version, "
                     "increment parlog version")
        stream = parlog_stream
        target = vdata
    else:
        logger.debug("next parlog version is above the data version, "
                     "increment data versions")
        stream = data_stream
        target = vparlog

    logger.debug("increment version for %s to %s" % (stream, target))
    versions = get_version_info(
        stream, None, timeout=timeout, protocol=protocol, useCache=False
    )["versionInfo"]
    max_version = 0  # it can't be zero at this point anyway
    if versions:
        max_version = max([x["number"] for x in versions])
    logger.debug("maximal created version for this stream is %d" % max_version)
    if max_version >= int(target[1:]):
        logger.debug("required target version is already created, use it")
        return target
    delta = int(target[1:]) - max_version
    logger.debug("create %d versions to reach the target" % delta)
    while delta > 0:
        delta -= 1
        _ = create_new_version(stream,
                               reason="sync version to %s" % target,
                               timeout=timeout)
    return target


def get_max_version(stream_name, timeout=1):
    """
        Returns maximal created version for the stream.

        This function returns version number as int. 0 - means no versions
        exist.
    """
    logger = logging.getLogger(__name__)
    logger.debug("get maximal created version for stream " + stream_name)
    protocol = "json"  # safe option, payload is minimal anyway
    # list all versions for the stream
    # note that one can do it with less data transmitted, but it's just more
    # convenient
    versions = get_version_info(
        stream_name, None, timeout=timeout, protocol=protocol,
        useCache=False
    )["versionInfo"]
    max_version = 0
    if versions:
        max_version = max([x["number"] for x in versions])
    logger.debug("maximal created version for this stream is %d" % max_version)
    return max_version


def create_target_version_for_shot(stream_name, shot, target_version,
                                   timeout=1):
    """
        Create target version for writing. If the target version is already
        used for the given shot, an error is raised.

        Parameters
        ----------
        stream_name: str
            Name of data of parlog stream.
        shot : str
            Program id in W7-X format.
        target_version: str
            Version of the form V1.

        Returns
        -------
        out: str
            Creates and returns target version if it is available for writing,
            raises an error otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.debug("create target version %s for stream %s" % (target_version,
                                                             stream_name))
    protocol = "json"  # safe option, payload is minimal anyway
    max_version = get_max_version(stream_name, timeout=timeout)
    last_version = get_last_version_for_program(
        stream_name, shot, timeout=timeout, protocol=protocol, useCache=False
    )
    last_version = 0 if last_version is None else int(last_version[1:])
    logger.debug("maximal used version for the shot %s is %d" % (shot,
                                                                 last_version))
    if target_version.__class__ in ["".__class__, u"".__class__]:
        target_version = int(target_version[1:])
    if last_version >= target_version:
        msg = ("Last used version (%d) for shot %s is above requested "
               "version %d" % (last_version, shot, target_version))
        logger.error(msg)
        raise RuntimeError(msg)
    elif target_version <= max_version:
        logger.debug("requested version was already created, return it")
        return "V" + str(target_version)
    else:
        delta = target_version - max_version
        logger.debug("target version %d is above max. created version "
                     "%d, create %d new versions" % (target_version,
                                                      max_version, delta))
        while delta > 0:
            delta -= 1
            _ = create_new_version(
                stream_name, reason="sync version to %s" % target_version,
                timeout=timeout
            )
        return "V" + str(target_version)


def create_target_sync_version_for_shot(stream_name, shot, target_version,
                                        timeout=1):
    """
        The same as create_target_version_for_shot, but keeps datastream and
        parlog versions synced.
    """
    logger = logging.getLogger(__name__)
    logger.debug("create synced target version %s for "
                 "stream %s" % (target_version, stream_name))

    base_stream = url._get_base_stream_name(stream_name)
    parlog_stream = base_stream + "_PARLOG"
    data_stream = base_stream + "_DATASTREAM"
    logger.debug("base stream name " + base_stream)
    logger.debug("parlog stream name " + parlog_stream)
    logger.debug("datastream stream name " + data_stream)
    create_target_version_for_shot(parlog_stream, shot, target_version,  timeout=timeout)
    return create_target_version_for_shot(data_stream, shot, target_version,  timeout=timeout)
