from archivedb import timing
from archivedb import programs
import numpy as np


def test_get_time_intervals(block_cache):
    signal = ("Test/raw/W7X/QTB/ADQ14DC_SPD-04381_DATASTREAM/"
              "0/channel_0")
    t1, t2 = 1457431564197103981, "2016-03-08 10:10:00"
    intervals = timing.get_time_intervals(signal, t1, t2,
                                          maxIntervals=10)
    assert intervals.shape == (10, 2)
    stored = np.loadtxt("tests/data/intervals_qtb_04381_channel_0_"
                        "20160308_max_10.txt")
    assert (intervals == stored).min()
    intervals = timing.get_time_intervals(signal, t1, t2,
                                          maxIntervals=10,
                                          protocol="cbor")
    assert intervals.shape == (10, 2)
    assert (intervals == stored).min()

def test_get_time_intervals_for_shot(block_cache):
    signal = ("Test/raw/W7X/QTB/ADQ14DC_SPD-04381_DATASTREAM/"
              "0/channel_0")
    shot = "20160308.010"
    t1, t2 = programs.get_program_from_to(shot)
    intervals1 = timing.get_time_intervals(signal, t1, t2)
    intervals2 = timing.get_time_intervals_for_program(signal, shot)
    assert (intervals1 == intervals2).min()
    intervals3 = timing.get_time_intervals_for_program(signal, shot,
                                                       useSingleInterval=True)
    assert (intervals3 == np.array([[t1, t2]])).min()
    # tstart is specified
    intervals1 = timing.get_time_intervals_for_program(signal, shot)
    intervals2 = timing.get_time_intervals_for_program(signal, shot,
                                                       tstart=-2)
    t1 = programs.get_program_t1(shot)
    i = intervals1[:,1] > (t1 - 2000000000)
    assert np.all(intervals1[i, :] == intervals2)
    # tstop is specified
    intervals1 = timing.get_time_intervals_for_program(signal, shot)
    intervals2 = timing.get_time_intervals_for_program(signal, shot,
                                                       tstop=5)
    t1 = programs.get_program_t1(shot)
    i = intervals1[:,1] < (t1 + 5000000000)
    assert np.all(intervals1[i, :] == intervals2)
    # tstart and tstop
    intervals1 = timing.get_time_intervals_for_program(signal, shot)
    intervals2 = timing.get_time_intervals_for_program(signal, shot,
                                                       tstart=-2,
                                                       tstop=5)
    t1 = programs.get_program_t1(shot)
    i = (intervals1[:,1] < (t1 + 5000000000)) & (intervals1[:,1] > (t1 - 2000000000))
    assert np.all(intervals1[i, :] == intervals2)


