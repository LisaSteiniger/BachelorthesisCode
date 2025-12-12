from archivedb import programs
from datetime import date, timedelta
import json


def test_is_cache_safe():
    assert programs._is_cache_safe("2018-10-16 00:00:01",
                                   "2018-10-16 23:59:59")

    today = date.today().strftime("%Y-%m-%d")
    assert not programs._is_cache_safe("2018-10-16 00:00:01",
                                       today + " 12:00:00")
    assert not programs._is_cache_safe(today + " 10:00:00",
                                       today + " 12:00:00")
    future = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert not programs._is_cache_safe("2018-10-16 00:00:01",
                                       future + " 12:00:00")
    assert not programs._is_cache_safe(future + " 10:00:00",
                                       future + " 12:00:00")

def test_is_cache_safe_for_id():
    assert programs._is_cache_safe_for_id("20181016.037")
    today = date.today().strftime("%Y%m%d")
    assert not programs._is_cache_safe_for_id(today + ".010")
    future = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    assert not programs._is_cache_safe_for_id(future + ".010")

def test_get_program_list(block_cache):
    tleft, tright = "2016-03-10 00:00:01", "2016-03-10 23:59:59"
    assert len(programs.get_program_list(tleft, tright)) == 40
    res = programs.get_program_list_for_day("2016-03-10")
    assert len(res) == 40
    fname = "tests/data/proginfo_20160310_0.json"
    with open(fname, "r") as f:
        stored = json.load(f)
    assert res[0] == stored
    res = programs.get_program_list_for_day("2016-03-10",
                                            protocol="cbor")
    assert len(res) == 40
    assert res[0] == stored

def test_get_program_id(block_cache):
    assert programs.get_program_id(1457599965437656661,
                                   protocol="cbor") == '20160310.001'
    assert programs.get_program_id(1457599965437656661,
                                   protocol="json") == '20160310.001'

def test_get_mdsplus_shot(block_cache):
    assert programs.get_mdsplus_shot(1457599966437656661) == 160310001

def test_get_program_t0(block_cache):
    assert programs.get_program_t0('20160310.001') == 1457599966437656661
    assert programs.get_program_t0('20160310.001',
                                   protocol="cbor") == 1457599966437656661

def test_get_program_from_to(block_cache):
    res = (1457600637990656661, 1457600704711656660)
    assert programs.get_program_from_to('20160310.002') == res
    assert programs.get_program_from_to('20160310.002',
                                        protocol="cbor") == res


def test_get_program_t1(block_cache):
    out = 1457599966437656661 + 60000000000
    assert programs.get_program_t1('20160310.001') == out
    assert programs.get_program_t1('20160310.001',
                                   protocol="cbor") == out


def test_parse_time_spec(block_cache):
    res = (1457600637990656661, 1457600704711656660)
    t1 = programs.get_program_t1("20160310.002")
    ts = "_".join([str(x) for x in res])
    assert programs.parse_time_spec('20160310.002',
                                    returnPlasmaStart=False) == res
    assert programs.parse_time_spec('20160310.002',
                                    returnPlasmaStart=False,
                                    timeout=10, protocol="cbor") == res
    assert programs.parse_time_spec('20160310.002') == res + (t1,)
    assert programs.parse_time_spec(ts,
                                    returnPlasmaStart=False) == res
    assert programs.parse_time_spec(ts) == res + (res[0],)
