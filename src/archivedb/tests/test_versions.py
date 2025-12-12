from archivedb import versions
from archivedb import programs
from archivedb import url
import logging
import json
import re
import pytest

def test_is_version_info_safe():
    assert versions._is_version_info_safe("mystream", 1)
    assert versions._is_version_info_safe("mystream", "v1")
    assert versions._is_version_info_safe("mystream", "V1")
    assert not versions._is_version_info_safe("mystream", None)

def test_get_version_information(block_cache):
    logging.getLogger(__name__).warning("Attention, versions "
                                        "can added to the datbase. "
                                        "Therefore, this test may "
                                        "not be strict.")
    fname = "tests/data/versioninfo_cbg_ecrh_total_power.json"
    with open(fname, "r") as f:
        stored = json.load(f)
    signal = "ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_PARLOG"
    res = versions.get_version_info(signal, None, timeout=10)
    assert "versionInfo" in res
    assert len(res["versionInfo"]) >= 3
    assert res["versionInfo"][-3:] == stored["versionInfo"]
    signal = "ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM"
    res = versions.get_version_info(signal, None, timeout=10)
    assert "versionInfo" in res
    assert len(res["versionInfo"]) >= 3
    for i in range(1, 4):
        for k in ['reason', 'analysis_environment', 'code_release',
                  'producer', 'number',]:
            # funny but creation tag changes ? how? why???
            assert res["versionInfo"][-i][k] == stored["versionInfo"][-i][k]

def test_get_last_version(block_cache):
    # unversioned stream
    # signal = ("ArchiveDB/raw/W7X/CoDaStationDesc.10082/"
              # "DataModuleDesc.10084_DATASTREAM/48/AAQ11_ActVal_I")
    # shot = "20160308.010"
    # assert versions.get_last_version_for_program(signal,
                                                 # shot, timeout=10) is None
    # t1, t2 = programs.get_program_from_to(shot)
    # assert versions.get_last_version(signal, t1, t2, timeout=10) is None
    # logging.getLogger(__name__).warning("Attention, versions "
                                        # "can added to the datbase. "
                                        # "Therefore, this test is "
                                        # "not strict.")
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/"
              "0/Ptot_ECRH")
    shot = "20181004.020"
    v = versions.get_last_version_for_program(signal, shot, timeout=10)
    assert isinstance(v, ("".__class__, u"".__class__))
    pattern = re.compile("^V[0-9]+$")
    assert pattern.match(v.upper()) is not None
    assert int(v[1:]) >= 1
    t1, t2 = programs.get_program_from_to(shot, timeout=10)
    v = versions.get_last_version(signal, t1, t2, timeout=10)
    assert isinstance(v, ("".__class__, u"".__class__))
    pattern = re.compile("^V[0-9]+$")
    assert pattern.match(v.upper()) is not None
    assert int(v[1:]) >= 1

def test_add_version_as_required(block_cache):
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/"
              "0/Ptot_ECRH")
    shot = "20181004.020"

    # nothing to be done
    assert versions.add_version_as_required(signal, shot, timeout=10) == (signal,
                                                              None)
    # ambiguous input
    with pytest.raises(RuntimeError):
        versions.add_version_as_required(signal, shot, version=1,
                                         timeout=10,
                                         useLastVersion=True)

    # explicit version
    ## channel without version
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/"
              "0/Ptot_ECRH")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/"
                             "TotalPower_DATASTREAM/V11/"
                             "0/Ptot_ECRH")
    res = versions.add_version_as_required(signal, shot, version=11)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="v11")
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="V11")
    assert res == (out, "V11")
    ## channel with version
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V7/"
              "0/Ptot_ECRH")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/"
                             "TotalPower_DATASTREAM/V11/"
                             "0/Ptot_ECRH")
    res = versions.add_version_as_required(signal, shot, version=11,
                                           updateURLVersion=True)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="v11",
                                           updateURLVersion=True)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="V11",
                                           updateURLVersion=True)
    assert res == (out, "V11")
    ## stream without version
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/"
                             "TotalPower_DATASTREAM/V11")
    res = versions.add_version_as_required(signal, shot, version=11)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="v11")
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="V11")
    assert res == (out, "V11")
    ## stream with version
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V7")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/"
                             "TotalPower_DATASTREAM/V11")
    res = versions.add_version_as_required(signal, shot, version=11,
                                           updateURLVersion=True)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="v11",
                                           updateURLVersion=True)
    assert res == (out, "V11")
    res = versions.add_version_as_required(signal, shot, version="V11",
                                           updateURLVersion=True)
    assert res == (out, "V11")
    ## view
    signal = ("ArchiveDB/views/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    res = versions.add_version_as_required(signal, shot, version=11)
    assert res == (signal, None)
    ## whatever
    signal = "aaa/bbb"
    res = versions.add_version_as_required(signal, shot, version=11)
    assert res == (signal, None)

    # no updateVersion
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V7")
    res = versions.add_version_as_required(signal, shot, version=2)
    assert res == (signal, "V7")
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V7/"
              "0/Ptot_ECRH")
    res = versions.add_version_as_required(signal, shot, version=2)
    assert res == (signal, "V7")

    # last version
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V1")
    res = versions.add_version_as_required(signal, shot,
                                           useLastVersion=True,
                                           timeout=10)
    assert res[0] >= out
    assert res[1] >= "V1"
    # unversioned stream
    # signal = ("ArchiveDB/raw/W7X/CoDaStationDesc.10082/"
              # "DataModuleDesc.10084_DATASTREAM/48/AAQ11_ActVal_I")
    # shot = "20171010.018"
    # res = versions.add_version_as_required(signal, shot,
                                           # useLastVersion=True,
                                           # timeout=10)
    # assert res == (signal, None)

@versions.versionize_signal
def dummy(signal, shot, t1, t2, key1=None, key2=None,
          **kwargs):
    return signal, shot, t1, t2, key1, key2, kwargs

def test_version_decorator(block_cache):
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    shot = "20171010.018"
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V2")
    out = (out, shot, 2, 3, "kk", "hh",
           {"version": None, "timeout":10, "returnVersion":None,
            "useLastVersion": None, "updateURLVersion": None,
            "returnSignalPath": None})
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                version=2, timeout=10)
    assert res == out
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V1")
    out = (out, shot, 2, 3, "kk", "hh")
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                useLastVersion=True,
                timeout=10)
    assert res[0] >= out[0]
    assert res[1:-1] == out[1:]
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V8")
    shot = "20171010.018"
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V2")
    out = (out, shot, 2, 3, "kk", "hh")
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                version=2, updateURLVersion=True)
    assert res[:-1] == out
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    shot = "20171010.018"
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V2")
    out = (out, shot, 2, 3, "kk", "hh")
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                version=2, returnVersion=True)
    assert res[0][:-1] == out
    assert res[1] == "V2"
    signal = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM")
    shot = "20171010.018"
    out = ("ArchiveDB/raw/W7X/CBG_ECRH/TotalPower_DATASTREAM/V2")
    out = (out, shot, 2, 3, "kk", "hh")
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                version=2, returnSignalPath=True)
    assert res[0][:-1] == out
    assert res[1] == signal + "/V2"
    res = dummy(signal, shot, 2, 3, key1="kk", key2="hh",
                version=2, returnSignalPath=True, returnVersion=True)
    assert res[0][:-1] == out
    assert res[1] == signal + "/V2"

