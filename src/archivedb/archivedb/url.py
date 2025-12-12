# -*- coding: utf-8 -*-
"""
    Set of functions to read/write to URL and to decode/encode data.
"""
try:
    import urllib2  # Python 2
    from urllib import quote, unquote  # Python 2
    from urlparse import urlparse
except ImportError:
    from urllib.parse import quote, unquote, urlparse  # Python 3
    import urllib.request as urllib2  # Python 3
import json
import cbor
import socket
import random
import logging
import numpy as np
from . import cache

# TODO
# 1. move reading png for images to here
# 2. add post request for adding data also here
# 4. add a better safe-context check, e.g. for signal requests
# it is safe, but not safe for version and interval requests
# 7. consider switching to requests

_db_url = "http://archive-webapi.ipp-hgw.mpg.de/"
_mdsplus_url = "http://10.44.4.11/operator/find_pulse/"

_block_requests = False  # for development purposes only


def _decode_compact_times(times_input):
    """
        Decoding of time stamps in compact format.
        Shamelessly copied from w7xarchive.
    """
    times_input = json.loads(times_input)
    times = [ int(times_input[0]) ]
    for spec in times_input[1:]:
        tmp = spec.split("x")
        length, step = int(tmp[0]), int(float(tmp[1]) * 1e9)
        times.extend(
            [ times[-1] + (i * step) for i in range(1, length + 1) ]
        )
    return np.array(times, dtype=np.int64)


def _decode_raw_images(response):
    """
        Decoding of raw images format. Shamelessly copied from w7xarchive.
    """
    # image metadata
    num_images = int(response.headers.get("X-Raw-NumImages", None))
    width = int(response.headers.get("X-Raw-Width", None))
    height = int(response.headers.get("X-Raw-Height", None))
    num_bytes = int(response.headers.get("X-Raw-NumBytes", None))
    times = response.headers.get("X-Raw-TimesCompact", None)
    if None in [num_images, width, height, num_bytes, times]:
        raise RuntimeError("Incomplete response header information for "
                           "raw images. One of X-Raw-NumImages, X-Raw-Width"
                           "X-Raw-Height, X-Raw-NumBytes"
                           "X-Raw-TimesCompact is missing in"
                           " %s" % str(response.headers))
    if num_bytes not in [1, 2, 4, 8]:
        raise RuntimeError("Number of bytes per pixel for the image %d "
                           "is not supported. Supported values are "
                           "1, 2, 4, 8" % num_bytes)

    # create a numpy array from the raw buffer, using proper type
    dtype = np.dtype(
        {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[num_bytes]
    ).newbyteorder(">")  # Archive uses big endian
    data = np.frombuffer(response.read(), dtype=dtype).reshape(
         (num_images, height, width)
     )
    return {"dimensions": _decode_compact_times(times), "values": data}


_decoder = {"json": json.loads, "cbor": cbor.loads,
            "raw_images": lambda x: x}
_encoder = {"json": json.dumps, "cbor": cbor.dumps}
_predecoder = {"json": lambda s: s.read().decode("utf-8"),
               "cbor": lambda s: s.read(),
                "raw_images": _decode_raw_images,
               }
_afterencoder = {"json": lambda s: s.encode("utf-8"), "cbor": lambda s: s}


def _get_ips_for_archive(url="archive-webapi.ipp-hgw.mpg.de", port=80,
                         family=socket.AF_INET, type_=socket.SOCK_STREAM,
                         num_attempts=20):
    logger = logging.getLogger(__name__)
    logger.info("resolve ip addresses for archive with %d "
                "retries" % num_attempts)
    ips = []
    for i in range(num_attempts):
        ips.extend(
            [
                x[-1][0] for x in
                socket.getaddrinfo(url, port, family=family, type=type_)
            ])
    ips = list(set(ips))
    logger.debug("found %d ips" % len(ips))
    return ips


_archive_ips = None
_archive_rotate_ip = False
try:
    _archive_ips = _get_ips_for_archive()
    random.shuffle(_archive_ips)
except:
    pass


@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.RAW,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="",
                           name_suffix=".txt",
                           hash_keywords=["protocol"],
                           cache_path=cache._cache_path + "rawdata/",
                           save_input_parameters=True,
                           validate_cache=cache._is_valid_input_string_in_cache)
def read_from_url_and_decode(url, timeout=1, retries=10, protocol="json",
                             **kwargs):
    """
        Read JSON/CBOR from URL with timeout and re-attempts.
    """
    logger = logging.getLogger(__name__)
    url = _make_full_url(url)
    logger.debug("requesting with protocol " + protocol + " url " + url)
    if _block_requests:
        logger.error("URL requests are blocked, raise an error.")
        raise RuntimeError("URL requests are blocked.")
    for i in range(retries):
        try:
            url_tmp = url
            if (_archive_rotate_ip and _archive_ips is not None and
                    (url[:len(_db_url)] == _db_url)):
                ip = _archive_ips[i % len(_archive_ips)]
                url_tmp = "http://" + ip + "/" + url[len(_db_url):]
            logger.debug("attempt url " + url_tmp)
            headers = {}
            if protocol in ["json", "cbor"]:
                headers = {"Accept": "application/%s" % protocol}
            request = urllib2.Request(url_tmp, headers=headers)
            response = urllib2.urlopen(request, timeout=timeout)
            return _decoder[protocol](_predecoder[protocol](response))
        except socket.timeout:
            logger.debug("url request timed out, iteration %d" % (i+1))
            continue
        except urllib2.HTTPError as ex:
            logger.exception("an HTTPError happened while reading url")
            raise RuntimeError(
                "HTTP error with the following message: %s" % ex.read())
    logger.info("%d attempts to read url %s failed" % (retries, url))
    raise RuntimeError("Failed to read signal from URL. "
                       "Possible reasons: wrong signal name, "
                       "data don't exist, too short timeout (try "
                       "timeout=10), network problems.")


def _make_string_signal_name(signal_name):
    """
        Converts a list form of the signal name into string,
        if required.
    """
    if not isinstance(signal_name, ("".__class__, u"".__class__)):
        return "/".join(str(c) for c in signal_name)
    return signal_name


def _make_full_url(signal_name):
    """
        Helper function to deal with URLs.
    """
    signal_name = _make_string_signal_name(signal_name)
    if (not signal_name.startswith(_db_url) and
            not signal_name.startswith(_mdsplus_url)):
        if not signal_name.startswith('/'):
            signal_name = _db_url + signal_name
        else:
            signal_name = _db_url[:-1] + signal_name
    url_ = urlparse(signal_name)
    path = quote(url_.path, safe="/%")
    query = "?" + url_.query if url_.query else ""
    return url_.scheme + "://" + url_.netloc + path + query
    # return signal_name
    # does url encoding of special characters
    # return quote(signal_name, safe=":/%")


def _get_only_signal_name(url):
    """
        Strips the server part from url, if applicable.
        Does not modify query ending! -> not an alternative to urlparse
    """
    if url.startswith(_db_url):
        return url[len(_db_url):]
    if url.startswith(_mdsplus_url):
        return url[len(_mdsplus_url):]
    return url


def _get_base_stream_name(stream):
    """
        Strip _DATASTREAM or _PARLOG part of the stream name, if present,
        and return the base stream name.

        Presently this does not support channeled names.
    """
    p1 = stream.rfind("_PARLOG")
    if p1 != -1:
        return stream[:p1]
    p2 = stream.rfind("_DATASTREAM")
    if p2 != -1:
        return stream[:p2]
    if stream.endswith("_"):
        return stream[:-1]
    return stream

