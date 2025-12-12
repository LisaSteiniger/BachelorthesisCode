from archivedb import signals
from archivedb import parlogs
import time
import numpy as np

def test_write_signal(block_cache):
    t0 = int(round(time.time() * 1000)) * 1000000
    t = [t0 + x for x in range(5)]
    s = [[11,2,3,4,15],[22,24,23,44,215]]
    signal = ("Test/raw/W7XAnalysis/webapi-tests/"
              "pythonTest1_DATASTREAM/")
    assert signals.write_signal(signal, t, s)
    t2, s2 = signals.get_signal(signal, t[0], t[-1])
    assert t2.shape[0] == 5
    assert s2.shape == (2, 5)
    assert (np.array(t) == t2).min()
    assert (np.array(s) == s2).min()
