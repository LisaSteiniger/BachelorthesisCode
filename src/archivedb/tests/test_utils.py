import numpy as np
from archivedb import utils

def test_to_timestamp():
    assert (utils.to_timestamp("2016-03-10 12:00:00") ==
            1457611200000000000)
    assert (utils.to_timestamp("2016-03-10 12:00:00.1213") ==
            1457611200121300000)
    assert (utils.to_timestamp(1457611200121300000) ==
            1457611200121300000)
    assert (utils.to_timestamp(np.int64(1)) == np.int64(1))

def test_to_stringdate():
    assert (utils.to_stringdate(1457611200000000000).split(".")[0] ==
            "2016-03-10 12:00:00")
    assert (utils.to_stringdate(1457599966437656661) ==
            "2016-03-10 08:52:46.437657")
