import numpy as np
from archivedb import url
import pytest

def test_make_full_url():
    x = ["Test", "raw", "W7X", "QTB_Profile", "volume_1_DATASTREAM",
         0, "Te_map"]
    url_ = ("http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X/"
            "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._make_full_url(x) == url_
    x = ("http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X/"
         "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._make_full_url(x) == url_
    x = "Test/raw/W7X/QTB_Profile/volume_1_DATASTREAM/0/Te_map"
    assert url._make_full_url(x) == url_

def test_read_from_url_and_decode(block_cache):
    url_ = ("http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/W7X"
            "/CBG_ECRH/TotalPower_DATASTREAM/V1/0/Ptot_ECRH/"
            "_signal.json?from=1538644768100793331"
            "&upto=1538644787100753331")
    fname = ("tests/data/CBG_ECRH_TotalPower_V1_0_Ptot_ECRH_"
             "1538644768100793331_1538644787100753331.json")
    rawdata = url.read_from_url_and_decode(url_, timeout=10)
    with open(fname, "rt") as f:
        storeddata = url.json.load(f)
    assert rawdata == storeddata[0]
    rawdata = url.read_from_url_and_decode(url_.replace("json", "cbor"),
                                           protocol="cbor", timeout=10)
    for k in ["dimensions", "label", "datatype", "unit",
              "dimensionSize", "sampleCount", "dimensionCount"]:
        assert rawdata[k] == storeddata[0][k]
    # OK, this is weird, but cbor returns slightly different values
    assert np.allclose(np.array(rawdata["values"]),
                       np.array(storeddata[0]["values"]))
    # example with forbidden characters in url?

def test_read_from_url_exceptions(block_cache):
    # wrong url
    url_ = ("http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/W7X"
            "/CBG_ECRH/TotalPower_DATASTREAM/VXXXXX/0/Ptot_ECRH/"
            "_signal.json?from=1538644768100793331"
            "&upto=1538644787100753331")
    with pytest.raises(RuntimeError):
        url.read_from_url_and_decode(url_)
    # too short timeout - request a huge data interval
    url_ = ("http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/W7X"
            "/CBG_ECRH/TotalPower_DATASTREAM/V1/0/Ptot_ECRH/"
            "_signal.json?from=1438644768100793331"
            "&upto=1638644787100753331")
    with pytest.raises(RuntimeError):
        url.read_from_url_and_decode(url_, timeout=1, retries=3)

def test_make_string_signal_name():
    x = ["Test", "raw", "W7X", "QTB_Profile", "volume_1_DATASTREAM",
         0, "Te_map"]
    url_ = ("Test/raw/W7X/"
            "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._make_string_signal_name(x) == url_
    x = ("http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X/"
         "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._make_string_signal_name(x) == x
    x = ("Test/raw/W7X/"
         "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._make_string_signal_name(x) == x

def test_get_only_signal_name():
    url_ = ("http://archive-webapi.ipp-hgw.mpg.de/Test/raw/W7X/"
            "QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    x = ("Test/raw/W7X/QTB_Profile/volume_1_DATASTREAM/0/Te_map")
    assert url._get_only_signal_name(url_) == x
    assert url._get_only_signal_name(x) == x


