from archivedb import parlogs
from archivedb import programs
import json

def test_get_parameters_box(block_cache):
    signal = ("ArchiveDB/raw/W7X/CoDaStationDesc.14823/"
              "DataModuleDesc.14833_PARLOG")
    shot = "20171018.020"
    par1 = parlogs.get_parameters_box_for_program(signal, shot)
    t1, t2 = programs.get_program_from_to(shot)
    par2 = parlogs.get_parameters_box(signal, t1, t2)
    assert par1 == par2
    par3 = parlogs.get_parameters_box_for_program(signal, shot,
                                                  protocol="cbor")
    assert par1 == par3
    with open("tests/data/parlog_20171018.020.json", "r") as f:
        stored = json.load(f)
    assert par1 == stored


