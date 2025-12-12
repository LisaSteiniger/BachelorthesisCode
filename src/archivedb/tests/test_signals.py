from archivedb import signals
from archivedb import programs
from archivedb import timing
import numpy as np

def test_get_signal(block_cache):
    signal = ("Test/raw/W7X/QTB_Central/volume_2_DATASTREAM/"
              "V1/1/ne_map")
    shot = "20171010.018"
    tleft, tright = programs.get_program_from_to(shot)
    t1, s1 = signals.get_signal(signal, tleft, tright)
    assert t1.shape == s1.shape
    assert t1.dtype == np.dtype(np.int64)
    t2, s2 = signals.get_signal(signal, tleft, tright,
                                protocol="cbor")
    assert (t1 == t2).min()
    assert np.allclose(s1, s2)
    t3, s3 = signals.get_signal_for_program(signal, shot,
                                            correctTime=False)
    assert (t1 == t3).min()
    assert (s1 == s3).min()
    intervals = timing.get_time_intervals_for_program(signal, shot)
    t4, s4 = signals.get_signal_multiinterval(signal, intervals)
    assert (t1 == t4).min()
    assert (s1 == s4).min()
    stored = np.load("tests/data/qtb_volume_2_nemap_20171010.018.npz")
    ts = stored["t"]
    ss = stored["s"]
    assert (t1 == ts).min()
    assert (s1 == ss).min()
    t5, s5 = signals.get_signal_for_program(signal, shot,
                                            correctTime=False,
                                            useSingleInterval=True)
    assert (t1 == t5).min()
    assert (s1 == s5).min()
    # tstart and tstop are provided
    t1, s1 = signals.get_signal_for_program(signal, shot)
    t2, s2 = signals.get_signal_for_program(signal, shot, tstart=-2,
                                            tstop=5)
    i = (t1 > -2) & (t1 < 5)
    assert np.all(s1[i] == s2)
    assert np.all(t1[i] == t2)


def test_get_signal_box(block_cache):
    signal = "Test/raw/W7X/QTB_Central/volume_2_DATASTREAM/V1"
    shot = "20171018.025"
    channels = [0, 1]
    tleft, tright = programs.get_program_from_to(shot)
    t1, s1 = signals.get_signal_box(signal, tleft, tright, channels)
    assert t1.shape[0] == s1.shape[1]
    assert s1.shape[0] == len(channels)
    assert t1.dtype == np.dtype(np.int64)
    t2, s2 = signals.get_signal_box(signal, tleft, tright, channels,
                                    protocol="cbor")
    assert (t1 == t2).min()
    assert np.allclose(s1, s2)
    t3, s3 = signals.get_signal_box_for_program(signal, shot, channels,
                                                correctTime=False)
    assert (t1 == t3).min()
    assert (s1 == s3).min()
    intervals = timing.get_time_intervals_for_program(signal + "/1/ne_map",
                                                      shot)
    t4, s4 = signals.get_signal_box_multiinterval(signal,
                                                  intervals, channels)
    assert (t1 == t4).min()
    assert (s1 == s4).min()

    stored = np.load("tests/data/qtb_volume_2_box_20171018.025.npz")
    ts = stored["t"]
    ss = stored["s"]
    assert (t1 == ts).min()
    assert (s1 == ss).min()
    t5, s5 = signals.get_signal_box_for_program(signal, shot, channels,
                                                correctTime=False,
                                                useSingleInterval=True)
    assert (t1 == t5).min()
    assert (s1 == s5).min()
    # tstart and tstop are provided
    t1, s1 = signals.get_signal_box_for_program(signal, shot, channels)
    t2, s2 = signals.get_signal_box_for_program(signal, shot, channels,
                                                tstart=-2, tstop=5)
    i = (t1 > -2) & (t1 < 5)
    assert np.all(s1[:, i] == s2)
    assert np.all(t1[i] == t2)


@signals._correct_time_to_plasma_start
def dummy(s, shot, t, correctTime=False):
    return t, s

def test_correct_time(block_cache):
    shot = "20171010.018"
    t0 = programs.get_program_t0(shot)
    s =  np.random.uniform(size=100)
    t = t0 + (60 + np.arange(100))*1000000000
    res = dummy(s, shot, t, correctTime=False)
    assert (res[0] == t).min()
    assert (res[1] == s).min()
    res = dummy(s, shot, t, correctTime=True)
    assert (res[0] == ((t-t0)*1e-9 - 60.0)).min()
    assert (res[1] == s).min()
