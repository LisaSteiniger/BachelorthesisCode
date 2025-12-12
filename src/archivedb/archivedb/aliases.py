"""
    Functions to resolve aliases.
"""
import numpy as np
import functools
import re
import logging
from . import url
from . import utils
from . import programs
from . import cache


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.ALIAS,
                           safe_context=lambda *args, **kwargs: False,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 100})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.ALIAS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="alias",
                           name_suffix=".txt")
def resolve_alias(alias, time_from, time_to, timeout=1,
                  cacheSettings=None, useCache=True):
    """
        Resolve alias and return a signal name.

        Parameters
        ----------
        alias : str
            Alias to resolve.
        time_from, time_to: int or str
            Time boundaries either as a date string or a nanosecond time stamp.

        timeout - see url.read_from_url_and_decode
        useCache, cacheSettings - control cache behaviour

        Returns
        -------
        out : str
            Signal name.
    """
    logger = logging.getLogger(__name__)
    logger.debug("resolve alias: " + alias + " " +
                 str(time_from) + " " + str(time_to))

    address = url._make_string_signal_name(alias)
    if address.strip("/").split("/")[1] != "views":
        logger.debug("this is not an alias, return original signal name")
        return alias

    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)
    address = (address +
               "?filterstart=%d&filterstop=%d" % (time_from, time_to))
    res = url.read_from_url_and_decode(address, timeout=timeout)
    key = "time_intervals"
    if (key in res and res[key] and "href" in res[key][0]):
        logger.debug("valid alias response")
        url_ = url.urlparse(res[key][0]["href"]).path.strip("/")
        url_ = url.unquote(url_)
        logger.debug("return " + url_)
        return url_
    else:
        logger.error("alias could not be resolved, server returned " +
                     url.json.dumps(res))
        raise RuntimeError("couldn't resolve alias")


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.ALIAS,
                           safe_context=lambda *args, **kwargs: False,
                           settings={"useMemCache": True,
                                     "maxCacheLength": 100})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.ALIAS,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="alias",
                           name_suffix=".txt")
def resolve_alias_for_program(alias, id, timeout=1,
                              cacheSettings=None, useCache=True):
    """
        Resolve alias and return a signal name.

        Parameters
        ----------
        alias : str
            Alias to resolve.
        id : str
            Program id.
        timeout - see url.read_from_url_and_decode
        useCache, cacheSettings - control cache behaviour

        Returns
        -------
        out : str
            Signal name.
    """
    logger = logging.getLogger(__name__)
    logger.debug("resolve alias " + alias + " for program " +
                 str(id))
    tleft, tright = programs.get_program_from_to(id, timeout=timeout,
                                                 protocol="json",
                                                 cacheSettings=cacheSettings,
                                                 useCache=useCache)
    return resolve_alias(alias, tleft, tright, timeout=timeout,
                         useCache=useCache, cacheSettings=cacheSettings)


_alias_check = re.compile("/views/") #  for rough, early checking


def convert_alias_to_signal(func):
    """
        Decorator for converting aliases to signals on the call stack
        of decorators.
    """
    @functools.wraps(func)
    def inner_function(*args, **kwargs):
        if (_alias_check.search(url._make_string_signal_name(args[0]))
                is not None):
            signal = args[0]
            id = args[1]
            timeout = kwargs.get("timeout", 1)
            useCache = kwargs.get("useCache", True)
            cacheSettings = kwargs.get("cacheSettings", None)
            signal = resolve_alias_for_program(signal, id, timeout=timeout,
                                               cacheSettings=cacheSettings,
                                               useCache=useCache)
            args = list(args)
            args[0] = signal
        return func(*args, **kwargs)
    return inner_function
