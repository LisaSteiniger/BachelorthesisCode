from archivedb import parlogs
import time

def test_write_parlog(block_cache):
    d = {'chanDescs': {'[0]':{'name' : 'Alpha', 'active' : 1,
                              'physicalQuantity' : { 'type' : 'X'} },
                       '[1]':{'name' : 'Zeta', 'active' : 1,
                              'physicalQuantity' : { 'type' : 'X'}} },
         'powerLevel' : 100 }
    name = "Test/raw/W7XAnalysis/webapi-tests/pythonTest178_PARLOG/"
    t = int(round(time.time() * 1000)) * 1000000
    res = parlogs.write_parameters(name, d, [t, -1])
    assert res
    d2 = parlogs.get_parameters_box(name, t, -1)
    assert d2["dimensions"] == [t, -1]
    assert d2["values"][0] == d
    t = int(round(time.time() * 1000)) * 1000000
    res = parlogs.write_parameters(name, d, [t, -1], protocol="cbor")
    assert res
    d2 = parlogs.get_parameters_box(name, t, -1)
    assert d2["dimensions"] == [t, -1]
    assert d2["values"][0] == d

def test_write_parlog_from_ini(block_cache):
    name = "Test/raw/W7XAnalysis/webapi-tests/"
    t = int(round(time.time() * 1000)) * 1000000
    assert parlogs.write_parameters_from_ini_file(name,
                                                  "tests/data/parlog.ini",
                                                   [t, -1])
    d = {'chanDescs': {'[0]' : {'unit': 'kg', 'active': 1,
                                'name': 'channel_0'},
                       '[1]' : {'unit': 'km', 'active': 1,
                                'name': 'channel_1'}},
         'Info': {'name': 'bztest', 'serial number': '1122'}}
    stream = name + "bztest_1122_PARLOG"
    d2 = parlogs.get_parameters_box(stream, t, -1)
    assert d2["dimensions"] == [t, -1]
    assert d2["values"][0] == d
