# -*- coding: utf-8 -*-
""" Caching for signals.

    Two types of caching are implemented: in memory and in file system.
    The in-memory cache is turned off/or set to a small number
    by default for signals to avoid bloating the library used in
    several python instances. For trigger information the in-memory cache
    is without the limits. The file system cache is on by default.

    The caching differentiates between safe and unsafe context. Safe means
    that the Archive entry does not change and the data can be considered
    immutable, e.g. for fixed version per time interval. The safe cache
    does not change and can be always used. The unsafe cache reading should
    be indicated either explicitly in cache settings argument or changed on
    the module level. In addition the unsafe cache will expire after given
    time and will be updated. It is also possible to force a cache update
    via settings.

    With workOnlyWithCache keyword one may turn off all safety checks and
    only work with data that are already in the cache. This can be useful
    for working offline.

    The cache system, both in-memory and in-file-system (POSIX only),
    is thread safe.  In addition, the file cache is multiprocess safe
    via using the fcntl module. This FS cache locking is not implemented for
    Wins.
"""
import numpy as np
import os
import functools
from contextlib import contextmanager
from threading import Lock
from collections import OrderedDict
import time
import json
import re
import hashlib
import logging
try:
    from urllib import quote, unquote  # Python 2
    from urlparse import urlparse
except ImportError:
    from urllib.parse import quote, unquote, urlparse  # Python 3
try:
    from fcntl import flock, LOCK_UN, LOCK_EX, LOCK_SH, LOCK_NB  # POSIX
except:
    def flock(fd, operation):  # a dummy
        pass
    LOCK_UN, LOCK_EX, LOCK_SH, LOCK_NB = range(4)  # need to support |
try:
    from inspect import signature, _empty # Python 3
except ImportError:
    try:
        from funcsigs import signature, _empty# Python 2, additional dependence
    except ImportError:
        # fallback if in py2 we don't have the dependency 
        _empty = None
        class Dummy(object):
            def __init__(self):
                self.parameters = {}

        def signature(*args, **kwargs):
            return Dummy()

from .version import __version__

# module settings
CACHE_FS, CACHE_MEM = list(range(2))
_cache_type_enc = {CACHE_FS: "FS cache", CACHE_MEM: "MEM cache"}
(SIGNAL, PARLOG, PROGLIST, VERSIONS, INTERVALS, IMAGE,
 ALIAS, EXTERNAL, RAW, ALIASEDSIGNAL) = list(range(0, 10))
_data_type_enc = {SIGNAL: "SIGNAL", PARLOG: "PARLOG", PROGLIST: "PROGLIST",
                  VERSIONS: "VERSIONS", INTERVALS: "INTERVALS",
                  IMAGE: "IMAGE", ALIAS: "ALIAS", EXTERNAL: "EXTERNAL",
                  RAW: "RAW", ALIASEDSIGNAL : "ALIASEDSIGNAL"}
_cache_settings = {"useFSCache": True,  # save to disk
                   "useMemCache": False,  # keep in memory - can bloat memory
                   "readUnsafeCache": False,
                   "writeUnsafeCache": True,
                   "cacheExpireTime": 36000,  # seconds, applies only to unsafe
                   "forceCacheUpdate": False,
                   "workOnlyWithCache": False,  # ignore reading rules, return
                   # only data from cache, e.g.
                   # for working offline
                   "maxCacheLength": 0,  # maximal length of cache
                   }
_block_cache = False  # for development - blocks any access to cache
_allow_pickle = False # needed for object arrays
# Note we use forward slashes throughout the code and use os.path.abspath
# to convert filenames before opening files. Forward slashes are natural
# for http and also look better in the in-memory cache.
_cache_path = os.path.expanduser(u"~/.archivedb/cache/")
__empty_object = object()  # sentinel for no cache
__archivedb_fingerprint = "archivedb v" + __version__


# TODO:
# 12. consider to have a single cache function with a list of sinks?
# 13. consider adding hashing and hash validation for too long
# file names


def _should_read_cache(settings, is_safe, cache_type):
    """
        Check whether to read from cache.
    """
    if _block_cache:
        logging.getLogger(__name__).warning("caching is blocked, this "
                                            "shouldn't happen in production")
        return False
    if settings["workOnlyWithCache"]:
        return True  # read data from any available source
    if settings["forceCacheUpdate"]:
        return False  # ignore available cache and proceed to read
    if not settings["readUnsafeCache"] and not is_safe:
        return False
    if settings["useFSCache"] and (cache_type == CACHE_FS):
        return True
    if settings["useMemCache"] and (cache_type == CACHE_MEM):
        return True
    return False  # how should I know what you are trying to do?


def _should_write_cache(settings, is_safe, cache_type):
    """
        Check whether to write to cache.
    """
    if _block_cache:
        logging.getLogger(__name__).warning("caching is blocked, this "
                                            "shouldn't happen in production")
        return False
    if settings["workOnlyWithCache"]:
        return False  # shouldn't overwrite with possibly empty values
    if not settings["writeUnsafeCache"] and not is_safe:
        return False
    if settings["useFSCache"] and (cache_type == CACHE_FS):
        return True
    if settings["useMemCache"] and (cache_type == CACHE_MEM):
        return True
    return False  # how should I know what you are trying to do?


def _tostring(x):
    if isinstance(x, np.ndarray):
        return repr(x.tolist()).replace(", ", ",")
    elif isinstance(x, list):
        return repr(x).replace(", ", ",")
    elif isinstance(x, ("".__class__, u"".__class__)):
        return str(x) # avoid additional quotes by repr
    return repr(x)


def _compose_cache_filename_for_signals(args, separator="_",
                                        prefix="", suffix="",
                                        path=_cache_path):
    """
        Compose cache file name for signal.

        args - is a list of input parameters, e.g. [signal, time_from, time_to],
        or [signal, shot_id]. Signal name is assumed to be the first.
    """
    # add = separator.join([str(x) for x in args[1:]])
    # Sometimes arguments are arrays instead of a string, which produces
    # ill names with \n signs. Also, lists produce extra white space after
    # comma, which is bad for long names. Therefore, use a small wrapper
    # to customize the stringification.
    add = separator.join([_tostring(x) for x in args[1:]])
    if prefix:
        prefix = prefix + separator
    signal = unquote(urlparse(args[0]).path)
    # signal = unquote(url._get_only_signal_name(args[0]))
    if signal and signal[0] == "/":
        signal = signal[1:]
    if not path and not signal:  # likely memcache
        return prefix + add + suffix  # only for better looking keys
    return path + signal + "/" + prefix + add + suffix


def _get_hash(args):
    """
        Simply hash all input arguments and return a string.
    """
    h = hashlib.sha512() # 128 symbols
    for arg in args:
        if isinstance(arg, np.ndarray):
            # ascontiguousarray will ensure the array is in C-order, i.e.
            # it will copy it if required. This is necessary, bcs. otherwise
            # hashlib fails with certain slices
            h.update(np.ascontiguousarray(arg))
        else:
            h.update(repr(arg).encode("utf-8"))
    h = h.hexdigest()
    return h


def _compose_hash_cache_filename(args, separator="_", prefix="",
                                 suffix="", path=_cache_path):
    """
        Compose cache file name with hashing instead of str.

        args - is a list of input parameters, e.g. [signal, time_from, time_to],
        or [signal, shot_id]. Signal name is assumed to be the first.
    """
    h = _get_hash(args)
    if prefix:
        prefix = prefix + separator
    if not path:  # likely memcache
        return prefix + h + suffix
    return path + "/" + prefix + h + suffix


def _compose_hash_cache_filename_with_signame(args, separator="_",
                                              prefix="", suffix="",
                                              path=_cache_path):
    """
        Compose cache file name preserving signal path, but hashing all the
        other arguments.  The signal path is assumed to be the first argument
    """
    add = _get_hash(args[1:])
    return _compose_cache_filename_for_signals([args[0], add],
                                               separator=separator,
                                               prefix=prefix, suffix=suffix,
                                               path=path)


def _compose_cache_filename_for_proginfo(args, **kwargs):
    """
        For program info requests url argument is absent, add empty string.
    """
    return _compose_cache_filename_for_signals([""] + list(args), **kwargs)


def _compose_cache_filename_for_raw(args, **kwargs):
    """
        For program info requests url argument is absent, add empty string.
    """
    url_ = args[0]
    if url_.startswith("https://"):
        url_ = url_[8:]
    if url_.startswith("http://"):
        url_ = url_[7:]
    # Query string can be very long, e.g. if many channels are requested for
    # a signal box. Therefore, we hash it.
    parts = url_.split("?")
    if len(parts) == 2:
        # if more ? are present, the format is unclear and we skip it
        url_ = parts[0] + "?" + _get_hash(parts[1:])
    return _compose_cache_filename_for_signals(["", url_] + args[1:], **kwargs)


def _check_cache_directory(fname):
    """Check that directory exists and create it if required.
    """
    # directory = fname[0: fname.rfind("/")]
    directory = os.path.dirname(fname)  # cross-platform
    logging.getLogger(__name__).debug("check directory exists " +
                                      directory)
    if not os.path.exists(directory):
        logging.getLogger(__name__).debug("creating cache path: " + directory)
        os.makedirs(directory)


def _correct_filename_for_win(fname):
    """
        Remove characters that make Bill unhappy.
    """
    if os.name in ["nt"] and fname:
        ind = fname.find(":\\")
        p1, p2 = "", fname
        if ind != -1:
            p1 = fname[:ind+2]
            p2 = fname[ind+2:]
        p2 = "".join(i if i not in ':<>"|?*' else "_" for i in p2)
        return p1 + p2
    return fname


@contextmanager
def _open_file_for_reading(fname, mode="b"):
    """
        Open file with shared flock for mthread/mprocess safety.
        This assumes that file exists.

        mode = b or t for binary and text
    """
    # on lock error we return so that the user can proceed with
    # reading signal from archive
    fname = _correct_filename_for_win(os.path.abspath(fname))
    _check_cache_directory(fname)
    logging.getLogger(__name__).debug("open file for reading " + fname)
    try:
        fd = open(fname, "r" + mode)
        flock(fd, LOCK_SH | LOCK_NB)
        yield fd
    finally:
        logging.getLogger(__name__).debug("close file " + fname)
        flock(fd, LOCK_UN)
        fd.close()


@contextmanager
def _open_file_for_writing(fname, mode="b"):
    """
        Open file with exclusive flock for mthread/mprocess safety.

        mode = b or t for binary and text
    """
    # on lock error we return. If cache is not updated it is not tragic.
    fname = _correct_filename_for_win(os.path.abspath(fname))
    _check_cache_directory(fname)
    logging.getLogger(__name__).debug("open file for writing " + fname)
    fd_new, fd_old = None, None  # seems to be required for wins
    try:
        if os.path.exists(fname):
            # file exists, we should not overwrite it before locking->
            # use a dummy read object to lock and than erase after
            # a successful locking
            fd_old = open(fname, "r" + mode)
            flock(fd_old, LOCK_EX | LOCK_NB)
            fd_new = open(fname, "w" + mode)
            yield fd_new
        else:
            fd_old = open(fname, "w" + mode)
            flock(fd_old, LOCK_EX | LOCK_NB)
            fd_new = None
            yield fd_old
    finally:
        logging.getLogger(__name__).debug("close file " + fname)
        flock(fd_old, LOCK_UN)
        if fd_old:  # otherwise fails in wins
            fd_old.close()
        if fd_new:
            fd_new.close()


def _put_signal_to_fscache(fname, res, max_cache_length):
    logging.getLogger(__name__).debug("save FS cache for " + fname)
    _check_cache_directory(fname)
    with _open_file_for_writing(fname, "b") as f:
        if isinstance(res[0], np.ndarray):
            np.savez(f, res[0], info=res[-1])
        else:
            np.savez(f, *res[0], info=res[-1])


def _get_signal_from_fscache(fname, empty_object):
    if not os.path.exists(fname):
        return empty_object
    with _open_file_for_reading(fname, "b") as f:
        tmp = dict(np.load(f, allow_pickle=_allow_pickle))  # avoid lazy reading
        n = len(tmp) - 1 # number of arrays in file, excluding info
        res = []
        for i in range(n):
            res.append(tmp["arr_%d" %i])
        if n == 1:
            return [res[0], tmp["info"]]
        else:
            return [tuple(res), tmp["info"]]



def _put_intervals_to_fscache(fname, res, max_cache_length):
    logging.getLogger(__name__).debug("save FS cache for " + fname)
    _check_cache_directory(fname)
    with _open_file_for_writing(fname, "b") as f:
        np.savez(f, intervals=res[0], info=res[-1])


def _get_intervals_from_fscache(fname, empty_object):
    if not os.path.exists(fname):
        return empty_object
    with _open_file_for_reading(fname, "b") as f:
        tmp = dict(np.load(f))  # avoid lazy reading
        return [tmp["intervals"], tmp["info"]]


def _put_json_to_fscache(fname, res, max_cache_length):
    logging.getLogger(__name__).debug("save FS cache for " + fname)
    _check_cache_directory(fname)
    with _open_file_for_writing(fname, "t") as f:
        json.dump(res, f)


def _get_json_from_fscache(fname, empty_object):
    if not os.path.exists(fname):
        return empty_object
    with _open_file_for_reading(fname, "t") as f:
        return json.load(f)


def _put_to_memcache(cache_storage, cache_lock,
                     fname, res, max_cache_length):
    logger = logging.getLogger(__name__)
    logger.debug("save MEM cache for " + fname)
    with cache_lock:
        cache_storage[fname] = res
        if max_cache_length:
            logger.debug("check MEM cache length")
            while (len(cache_storage) and
                   len(cache_storage) > max_cache_length):
                # need to clean up
                k = list(cache_storage.keys())[0]
                logger.debug("MEM cache length exceeded, remove " + k)
                del cache_storage[k]
            logger.debug("new MEM cache length is %d" % len(cache_storage))


def _get_from_memcache(cache_storage, cache_lock,
                       fname, empty_object):
    with cache_lock:
        v = cache_storage.get(fname, empty_object)
        # if element present, relocate it as last accessed
        if v is not empty_object:
            cache_storage.move_to_end(fname)
        return v


# check cache hasn't expired yet
def _is_not_too_old(res, is_safe, settings, hash_args, *args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.debug("check cache hasn't expired")
    # time since cache entry was created
    dt = np.abs(int(time.time()) - int(res[-1][-1]))
    if (is_safe or dt <= settings["cacheExpireTime"] or
            settings["workOnlyWithCache"]):
        logger.debug("cache is safe, or not too old or working "
                     " only with cache")
        return True
    logger.debug("ignore cache, bcs. it has expired or is unsafe")
    return False


# check aliased signal path, assuming path is passed as path
def _is_valid_aliasedsignal(res, is_safe, settings, hash_args, *args,
                            **kwargs):
    logger = logging.getLogger(__name__)
    logger.debug("check if cache is valid for an aliased signal")
    # assume path is -2 in the info part
    if len(res) != 2 or len(res[-1]) < 2:
        logger.debug("info part of the cache entry is too short")
        return False
    if str(res[-1][-2]) == kwargs.get("path", ""):
        logger.debug("stored path matches the request, check age")
        return _is_not_too_old(res, is_safe, settings, hash_args, *args,
                               **kwargs)
    logger.debug("cache path does not appear to be valid")
    return False


# check saved input parameters match the passed ones (string version)
def _is_valid_input_string_in_cache(res, is_safe, settings, hash_args,
                                    *args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.debug("check if input parameters in cache match the call (strings)")
    # assume string version is -3 in the info part
    if len(res) != 2 or len(res[-1]) < 3:
        logger.debug("info part of the cache entry is too short")
        return False
    logger.debug("convert input parameters to string")
    input_string = "_".join([_tostring(x) for x in hash_args])
    if input_string == res[-1][-3]:
        logger.debug("stored input matches the request, check age")
        return _is_not_too_old(res, is_safe, settings, hash_args, *args,
                               **kwargs)
    logger.debug("cache input does not appear to be valid")
    return False


_views_re = re.compile("/views/")
_version_re = re.compile("/V[0-9]+(/|$)")


def _is_aliasedsignal_safe(*args, **kwargs):
    """
        Check that aliased signal is safe. Safe means a direct
        signal path without views and with version is passed as
        a path keyword.
    """
    path = kwargs.get("path", None)
    if (path is None or _views_re.search(path) is not None or
            _version_re.search(path) is None):
        return False
    return True


# This is a decorator that can be applied to do all types of caching
# for different types of data. The top level function chooses appropriate
# cache writing/reading routines and returns the actual decorator that wraps
# the input function by adding read/write statements. The configuration
# of cache types is done by arguments to the top level function. At the run
# time it is possible to tune which types of cache are used by using
# the cacheSettings key word to archive functions, global _cache_settings
# in this module or __memcache_settings__ and __fscache_settings__ arguments
# of the function. As for the mem cache, the main differences to the
# standard lru_cache are: thread safe, custom key composition, ignoring
# some technical keywords, control of settings and of safe context,
# cache expiration by time.
def cache_this_function(cache_type=CACHE_FS, data_type=SIGNAL,
                        name_prefix="", name_suffix="",
                        safe_context=lambda *args, **kwargs: False,
                        compose_cache_filename=_compose_cache_filename_for_signals,
                        put_to_cache=None, get_from_cache=None,
                        should_read_cache=_should_read_cache,
                        should_write_cache=_should_write_cache,
                        validate_cache=_is_not_too_old,
                        settings=None, hash_keywords=None,
                        cache_path=_cache_path,
                        save_input_parameters=False,
                        convert_inputs_to_string=True,
                        additional_information=None):
    """
        Decorator for caching a generic archivedb function.

        For standard data types, all the functions will be set internally.
        The same is true for the external data and in-memory cache.
        For external fs-cache, please supply at least put_to_cache
        and get_from_cache (e.g. use available here for signals, intervals
        or generic json).

        Parameters
        ----------
        cache_type : enum - CACHE_FS or CACHE_MEM
            Cache type FS - file system, MEM - in memory
        data_type : enum - SIGNAL, PARLOG, etc.
            Type of data to cache. For usage outside of the library use
            EXTERNAL.
        safe_context : callable
            A function that gets all input args and kwargs of the function to
            cache and should return True for safe context and False othws.
        name_prefix : str
            File name/key prefix.
        name_suffix : str
            File name/key suffix, e.g. file extension as '.npz'
        compose_cache_filename : callable
            Function that can compose cache name. It should accept:
            a list of arguments used to hash, separator, prefix, suffix and
            cache path. The return should be a string with full path.
        put_to_cache : callable
            Function for saving in cache. Should accept: fname (str),
            result to store (a list [actual result, [..., creation timestamp]]),
            maximal length of cache (int, can be ignored). The actual
            result can be a tuple or an array, depending on the function return.
            The second part contains an information list that contains at
            least the creation date, but is likely to be expanded.
        get_from_cache : callable
            Function for retrieving from cache. Should accept: fname (str),
            empty object (return on missing key); should return
            [result, [..., creation timestamp]]. The result should be
            returned in exactly the same form as passed to put_to_cache.
        should_read_cache, should_write_cache : callable
            Should return True/False, input includes settings dictionart,
            is_safe boolean and cache type.
        validate_cache : callable
            Function that checks that read cache is valid and should
            be returned. It should accepts: storeed cache result,
            is_safe context flag, cache settings dictionary, list of
            hash arguments (i.e. including keywords to hash), all
            input arguments and keyword arguments *args, **kwargs.
        settings : dict
            Settings to apply to this function. This overwrites the globals.
            They are exposed as __memcache_settings__ or __fscache_settings__.
        hash_keywords : list
            Usually cache name is composed based only on the non-keyword
            arguments. If required, pass keyword argument names that should
            be included into the hash as this list.
        cache_path : str
            Top level file system path to store FS cache.
        save_input_parameters : boolean
            If True, a representation of input parameters will
            be saved into the cache file and can be later checked with
            a cache validator. Input parameters are added into info part
            at the -3 location: [res, [addon, hash_args, path, timestampt]].
            hash_args are input arguments in the order of positional arguments and continued
            with keyword arguments, as they are mentioned in the input to
            the decorator.
        convert_inputs_to_string : boolean
            If True, input parameters are converted to string with _
            as separator. This avoids usage of allow_pickle. "repr" is used
            for the conversion, so that in most cases this should be
            sufficiently good.
        addittional_information : list
            A list of additional information tags that is written into
            a cache file. By default, the additional saved information
            includes the current archivedb versio and the creation time stamp
            of a cached value. Here, one could pass additional tags that
            should be stored, e.g. an external library version tag.
            Note, however, that presently this information is not
            returned in any form.
    """
    # This is a static per function setting dictionary, that can be modified
    # at run time (exposed via function attributes).
    cache_settings = {}  # empty by default, so that global options work
    if settings:
        cache_settings.update(settings)
    if cache_path and cache_path[-1] != "/":
        cache_path = cache_path + "/"
    # This selector simplifies use of the decorator in the library:
    # it is sufficient to pass only signal type and cache type
    if cache_type == CACHE_FS and data_type in [SIGNAL, IMAGE,
                                                ALIASEDSIGNAL]:
        # This is signal caching in the file system, with file locking.
        put_to_cache = _put_signal_to_fscache
        get_from_cache = _get_signal_from_fscache
    elif cache_type == CACHE_FS and data_type == INTERVALS:
        put_to_cache = _put_intervals_to_fscache
        get_from_cache = _get_intervals_from_fscache
    elif cache_type == CACHE_FS and data_type in [PROGLIST, VERSIONS,
                                                  PARLOG, ALIAS, RAW]:
        # caching of program lists
        put_to_cache = _put_json_to_fscache
        get_from_cache = _get_json_from_fscache
    elif cache_type == CACHE_MEM:
        # this is generic and can be used for any data, including external
        cache_path = ""
        cache_storage = OrderedDict()  # remembers order of insertion
        cache_lock = Lock()
        # wrap the actual functions, providing them with newly created
        # storage and lock

        def put_to_cache(fname, res, max_cache_length):
            _put_to_memcache(cache_storage, cache_lock, fname, res,
                             max_cache_length)

        def get_from_cache(fname, empty_object):
            return _get_from_memcache(cache_storage, cache_lock, fname,
                                      empty_object)
    if data_type == PROGLIST:
        compose_cache_filename = _compose_cache_filename_for_proginfo
    if data_type == RAW:
        compose_cache_filename = _compose_cache_filename_for_raw
    if data_type == ALIASEDSIGNAL:
        validate_cache = _is_valid_aliasedsignal
    # if any of required functions remains unset, e.g. with EXTERNAL,
    # raise early
    if None in [put_to_cache, get_from_cache, should_read_cache,
                should_write_cache, compose_cache_filename,
                validate_cache]:
        raise RuntimeError("Not all required input functions are set.")
    # prepare additional information list
    addon = [__archivedb_fingerprint]
    if additional_information is not None:
        addon = addon + additional_information
    cache_attr = "__fscache_settings__"
    if (cache_type == CACHE_MEM):
        cache_attr = "__memcache_settings__"
    # This is the actual decorator that uses the settings prepared above.
    logger = logging.getLogger(__name__)
    def inner_decorator(func):
        #inspect function signature and populate default values
        #of keyword arguments to be hashed
        default_keyvalues = {}
        if hash_keywords:
            logger.debug("inspect function signature and find default values")
            s = signature(func).parameters
            for k in hash_keywords:
                logger.debug("handle key " + k)
                if k in s:
                    v = s[k].default
                    default_keyvalues[k] = v if v is not _empty else None
                else:
                    logger.info("keyword " + k + "is no in the signature")
                    default_keyvalues[k] = None
            if logger.isEnabledFor(logging.DEBUG):  # avoid json generation
                logger.debug("default keyword values: " +
                             json.dumps(default_keyvalues))
        # the standard wrapper to ensure func attributes
        @functools.wraps(func)
        def cache(*args, **kwargs):
            logger.info(_cache_type_enc[cache_type] + " for function " +
                         func.__name__)

            # get cache settings from all sources
            settings = dict(_cache_settings)  # first global
            # per function
            settings.update(cache_settings)
            # input to the function call
            settings_in = kwargs.get("cacheSettings", None)
            # the keyword can be present, but set to None
            settings_in = settings_in if settings_in is not None else {}
            settings.update(settings_in)
            # old useCache keyword is interpreted as use FS cache
            if "useCache" in kwargs:
                settings["useFSCache"] = kwargs["useCache"]
            if logger.isEnabledFor(logging.DEBUG):  # avoid json generation
                logger.debug("cache settings " + json.dumps(settings))
            # these two settings are relevant only for an external use
            # of aliased signals that should be cached before versionizer
            returnSignalPath = kwargs.get("returnSignalPath", False)
            signalPath = ""
            if (data_type in [SIGNAL, PARLOG, IMAGE, INTERVALS]):
                # try to store reasonable path
                if (isinstance(args[0], ("".__class__, u"".__class__)) and
                        (args[0].find("_DATASTREAM") != -1 or
                         args[0].find("_PARLOG") != -1)):
                    signalPath = args[0]
                else:
                    signalPath = kwargs.get("path", "") # typical use case

            # create cache key/file name
            # assume that signal name is always the first and that
            # all other non-keyword arguments are times and shot numbers
            # keyword arguments are expected as keywords, othws. use bind?
            hash_args = args
            if hash_keywords:
                hash_args = list(args)
                for k in hash_keywords:
                    hash_args.extend([k, kwargs.get(k,
                                                    default_keyvalues[k])])
            fname = compose_cache_filename(hash_args, prefix=name_prefix,
                                           suffix=name_suffix, path=cache_path)
            logger.debug("cache key " + fname)

            # find if context is safe and whether to read cache
            is_safe = safe_context(*args, **kwargs)
            logger.debug("safe context is " + str(is_safe))
            if should_read_cache(settings, is_safe, cache_type):
                logger.debug("try to read cache")
                res = __empty_object
                try:
                    res = get_from_cache(fname, __empty_object)
                except:
                    logger.exception("failed to read cache " + fname)
                if (res is not __empty_object and
                        validate_cache(res, is_safe, settings,
                                       hash_args, *args, **kwargs)):
                    logger.debug("found valid cache for " + fname)
                    if data_type == ALIASEDSIGNAL and returnSignalPath:
                        return res[0], res[-1][-2]
                    else:
                        return res[0]
                else:
                    logger.debug("no cache or cache is invalid")
            else:
                logger.debug("cache reading is off for this context")

            # This exception block does not work, because top level cached
            # functions need to call other cached functions. As a result,
            # there might be no request to URL, but we need to call the inner
            # function. Is there a more clever way?
            # if settings["workOnlyWithCache"]:
                # logger.warn("Working in cache only mode but no cached data "
                # "available. Raise an exception")
                # raise RuntimeError("No cache available in cache only mode.")
                # return None # ?

            # proceed and call the actual function
            # finding the last version will not happen here any more,
            # bcs. it is now handled by a special decorator on the top of the
            # stack. The data are saved already per version.
            logging.debug("call inner function")
            timestamp = int(time.time())  # when the cache was created
            if data_type == ALIASEDSIGNAL: # keep resolved alias for validation
                kwargs["returnSignalPath"] = True
                res, signalPath = func(*args, **kwargs)
            else:
                res = func(*args, **kwargs)

            # write/update cache
            if should_write_cache(settings, is_safe, cache_type):
                logger.debug("write cache for " + fname)
                try:
                    if save_input_parameters:
                        logger.debug("branch with saving input parameters")
                        if convert_inputs_to_string:
                            logger.debug("convert input parameters to string")
                            hash_args = "_".join([_tostring(x) for x
                                                  in hash_args])
                        put_to_cache(fname, [res, addon +
                                             [hash_args, signalPath,
                                              timestamp, ]],
                                     settings.get("maxCacheLength", 0))
                    else:
                        put_to_cache(fname, [res, addon + [signalPath,
                                                           timestamp, ]],
                                     settings.get("maxCacheLength", 0))
                except:
                    logger.exception("Failed to write to cache with "
                                     "key/filename = " + fname)
            else:
                logger.debug("cache writing is off for this context")
            logger.debug("end of cache wrapper " +
                         _cache_type_enc[cache_type] + " for function " +
                         func.__name__)
            if data_type is ALIASEDSIGNAL and returnSignalPath:
                return res, signalPath # for external use above versionizer
            return res
        if cache_type == CACHE_MEM:
            cache.__memcache_storage__ = cache_storage
            cache.__memcache_lock__ = cache_lock
        setattr(cache, cache_attr, cache_settings)
        return cache
    return inner_decorator
