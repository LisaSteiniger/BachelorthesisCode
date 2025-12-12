# -*- coding: utf-8 -*-
"""
    Set of functions dealing with image data.
"""
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import socket
import tables
import random
from . import cache
from . import url
from . import utils
from . import timing
from . import parlogs


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.IMAGE,
                           safe_context=lambda *args, **kwargs: True)
# settings={"useMemCache": False,
# "maxCacheLength": 0})
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.IMAGE,
                           safe_context=lambda *args, **kwargs: True,
                           name_prefix="imagepng",
                           name_suffix=".npz")
def get_image_png(signal_name, time_from, time_to, timeout=1, retries=10,
                  cacheSettings=None, useCache=True):
    """
        Returns the last image within a given interval.

        The image is returned as a 2d array. This function uses png-format
        for data transfer from the data base.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        time_from, time_to: int or string
            Time boundaries of the form 'YYYY-MM-DD HH:MM:SS.%f',
            or nanosecond time stamp.

        Returns
        -------
        time stamp, image : int, 2d numpy array
            Time stamp of the image and 2d data array.

        Examples
        --------
        >>> get_image_png('Test/raw/Data4SoftwareTest/Hdf5.video.Test2/pixelfly_DATASTREAM',
                          1421229109876999999,
                          1421229109967000000)
    """
    logger = logging.getLogger(__name__)
    logger.debug("get image data")

    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)

    baseaddress = url._make_full_url(signal_name)
    baseaddress = baseaddress + "/_signal.png"
    address = (baseaddress+'?from=' + str(int(time_from)) +
               '&upto=' + str(int(time_to)))

    # why is this not using the standard url method?
    # can the standard be expanded to used png as well?
    request = url.urllib2.Request(address)
    response = None
    logger.debug("read data from url " + address)
    for i in range(retries):
        try:
            response = url.urllib2.urlopen(request, timeout=timeout)
            logger.debug("successfully downloaded data")
            break
        except socket.timeout:
            logger.debug("socket timeout, try again")
            continue
    if response is None:
        logger.error("failed to read data, raise exception")
        raise RuntimeError("Failed to download data. Possible reasons: "
                           "signal doesn't exist, too short timeout, "
                           " network/server problems.")

    logger.debug("convert data and return")
    img = Image.open(BytesIO(response.read()))

    pixelarray = np.array(img.getdata()).reshape(img.size[1], img.size[0])

    y = int(response.info()["W7X-Timestamp"])

    del request, response
    return y, pixelarray

# Don't cache this one, bcs. it will use the above function which is cached.


def get_image_png_multiple(signal_name, time_from, time_to, timeout=1,
                           cacheSettings=None, useCache=True):
    """
        Returns the all images within a given interval.

        The images are returned as a 3d array. This function uses png for
        data transfer. Multiple calls to get_image_png are employed.

        Note: the images are sorted from the oldest to the newest.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        time_from, time_to: int or string
            Time boundries of the form 'YYYY-MM-DD HH:MM:SS.%f',
            or nanosecond time stamp.

        Returns
        -------
        time stamp, image : array, 3d numpy array
            Time stamps of the images and 3d data array.

        Examples
        --------
        >>> get_image_png('Test/raw/Data4SoftwareTest/Hdf5.video.Test2/pixelfly_DATASTREAM',
                          1421229109876999999,1421229109967000000)
    """
    logger = logging.getLogger(__name__)
    logger.debug("read multiple images in png format")
    times = []
    data = []
    logger.debug("find available time intervals")
    intervals = timing.get_time_intervals(signal_name, time_from, time_to)
    # logger.debug("found time intervals " + url.json.dumps(intervals.tolist()))
    for interval in intervals:
        x, y = get_image_png(signal_name, interval[0], interval[1],
                             timeout=timeout,
                             useCache=useCache, cacheSettings=cacheSettings)
        times.append(x)
        data.append(y)
    return np.array(times), np.array(data)


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.IMAGE,
                           safe_context=lambda *args, **kwargs: False)
@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.IMAGE,
                           safe_context=lambda *args, **kwargs: False,
                           name_prefix="imagejson",
                           name_suffix=".npz")
def get_image(signal_name, time_from, time_to,
              correct_negatives=True, timeout=10, protocol="json",
              cacheSettings=None, useCache=False):
    """
        Returns a set of images as a 3d array.

        This function can use CBOR, JSON or RAW (fastest?) for data transfer.
        See the protocol keyword.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        time_from, time_to: int or string
            Time boundaries of the form 'YYYY-MM-DD HH:MM:SS.%f',
            or nanosecond time stamp.
        correct_negatives : bool, optional
            Whether to correct negative values. This is sometimes required, because
            the actual unsigned data are represented as data with sign. Not
            applied for the RAW protocol.
        protocol: str
            Which protocol to use for data transfer. Possible values are: json,
            cbor, raw. Note that url layer caching is disabled for some of the
            protocols.
        timeout : int, optional
            URL timeout
        Returns
        -------
        time stamp, image : int, 2d numpy array
            Time stamps of the images and 3d data array.

        See Also
        --------
        get_image_png : Returns the last image as a 2d array.

        Examples
        --------
        >>> get_image('Test/raw/Data4SoftwareTest/Hdf5.video.Test2/pixelfly_DATASTREAM',
                      1421229109876999999,1421229109967000000)
    """
    logger = logging.getLogger(__name__)
    logger.debug("read images in %s format" % protocol)

    time_from = utils.to_timestamp(time_from)
    time_to = utils.to_timestamp(time_to)

    baseaddress = (
        url._make_string_signal_name(signal_name) + "/_signal.%s" %protocol
    )
    address = (
        baseaddress +
        '?from=' + str(int(time_from)) +
        '&upto=' + str(int(time_to))
    )

    if protocol == "raw":
        image_data = url.read_from_url_and_decode(
            address, timeout=timeout, protocol="raw_images", useCache=False,
            cacheSettings={"useFSCache": False}
        )
        times = image_data["dimensions"]
        data = image_data["values"]
    else:
        image_data = url.read_from_url_and_decode(address, timeout=timeout,
                                                  protocol=protocol)
        data = np.array(image_data["values"])
        if correct_negatives:
            logger.debug("correct negative values in the data")
            data[data < 0] = data[data < 0] + 32768 + 32768
        times = np.array(image_data["dimensions"])
    return times, data


def write_image_to_database(signal_name, timestamp, im, parameters,
                            camname="pixelfly"):
    """
        Writes image, time stamp and parameters (PARLOG) to the database.

        Parameters
        ----------
        signal_name : string or list
            The signal address.
        timestamp : int
            Time in nanoseconds when the image was taken
        im : nd.array
            A 2d array of the image.
        parameters : dict
            Example : d =  {"projectName":"test data", "cameraType":"test data",
                            "height" : 287,
                            "width" : 380,
                            "bitDepth" : 10,
                            "dataBoxSize" : 1,
                            "datatype" : "short",
                            "unsigned" : 1}
        camname : str
            Camera name, used to create the full signal name.
        Returns
        -------
        out : boolean
            True if the image is successfully written.

        Examples
        --------
        >>> im = PIL.Image.open("w7x.jpg")
            width, height = im.size
            image = np.array(im.getdata()).reshape(height, width, 3).sum(axis=-1)
            d =  {"projectName":"test data", "cameraType":"test data",
            "height" : 287,
            "width" : 380,
            "bitDepth" : 10,
            "dataBoxSize" : 1,
            "datatype" : "short",
            "unsigned" : 1}
            write_image_to_database("Test/raw/W7XAnalysis/HDF5_import_test/",
                                    utils.to_timestamp("2015-08-03 12:30:00"),
                                    image, d)
    """
    # improve me :(
    logger = logging.getLogger(__name__)
    logger.info("write image to the database")
    baseaddress = url._make_full_url(signal_name)

    logger.debug("create temporary hdf5 file and populate it")
    randomname = str(random.randint(0, 100))+".h5"
    height, width = im.shape
    image = np.zeros((height, width, 1), int)
    image[..., 0] = im
    h5file = tables.open_file(randomname, "a", driver="H5FD_CORE",
                              driver_core_backing_store=0)
    resp = 0
    try:
        h5file.create_group("/", "data")
        h5file.create_array("/data", "timestamps",
                            obj=np.array([timestamp]))

        h5file.create_array("/data", camname, obj=image.astype(np.uint16))
        h5file.create_group("/", camname)

        h5file.create_group("/" + camname, "parms")

        i = 0
        pk = list(parameters.keys())
        pv = list(parameters.values())
        while i != len(pk):
            try:
                h5file.create_array("/" + camname + "/parms",
                                    pk[i], obj=pv[i])
            except Exception:
                h5file.create_array("/" + camname + "/parms",
                                    pk[i], obj=pv[i].encode())
            i = i+1

        h5file.create_group("/" + camname + "/parms",
                            "chanDescs")
        for x in h5file.get_node("/" + camname + "/parms"):
            if isinstance(x, tables.Group):
                continue
            del x.attrs.TITLE
            del x.attrs.FLAVOR
            del x.attrs.CLASS
            del x.attrs.VERSION

        h5file.create_group("/" + camname + "/parms/chanDescs",
                            "[0]")
        try:
            h5file.create_array("/" + camname + "/parms/chanDescs/[0]",
                                "name", obj="image")
        except Exception:
            h5file.create_array("/" + camname + "/parms/chanDescs/[0]",
                                "name", obj="image".encode())
        h5file.create_array("/" + camname + "/parms/chanDescs/[0]",
                            "active", obj=np.int32(1))
        for x in h5file.get_node("/" + camname + "/parms/chanDescs/[0]"):
            del x.attrs.TITLE
            del x.attrs.FLAVOR
            del x.attrs.CLASS
            del x.attrs.VERSION

        path_query = ('?dataPath=/data/' + camname +
                      '&timePath=/data/timestamps' +
                      '&parameterPath=/' + camname + '/parms')
        buf = h5file.get_file_image()
        req = url.urllib2.Request(url=baseaddress + path_query, data=buf,
                                  headers={'Content-Type': 'application/x-hdf',
                                           'Content-Length': '%d' % len(buf)})
        try:
            logger.info("send data")
            resp = url.urllib2.urlopen(req).code
        except url.urllib2.HTTPError as ex:
            logger.exception("Error while sending data")
            msg = ex.read()
            raise RuntimeError(msg)
    finally:
        h5file.close()
    logger.info("finished uploading image")
    return resp == 201
