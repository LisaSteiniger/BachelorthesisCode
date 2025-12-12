from archivedb import images
import numpy as np

def test_get_image_png(block_cache):
    signal = ("Test/raw/Data4SoftwareTest/Hdf5.video.Test2/"
              "pixelfly_DATASTREAM")
    t1, t2 = 1421229109876999999, 1421229109967000000
    t, d = images.get_image_png(signal, t1, t2)
    assert d.shape == (1392, 1024)
    stored = np.load("tests/data/image_png_1.npz")
    ts = stored["t"]
    ds = stored["d"]
    assert t == ts
    assert (d == ds).min()

def test_get_image(block_cache):
    signal = ("ArchiveDB/raw/W7X/QSR_Limiter_NIR/"
              "AEF10_A_Port_DATASTREAM")
    t1, t2 = "2016-03-10 15:00:29.9", "2016-03-10 15:00:30"
    t, d = images.get_image(signal, t1, t2)
    assert t.shape[0] == 5
    assert d.shape == (5, 288, 720)
    stored = np.load("tests/data/image_json_1.npz")
    ts = stored["t"]
    ds = stored["d"]
    assert (t == ts).min()
    assert (d == ds).min()

def test_get_image_png_multiple(block_cache):
    signal = ("ArchiveDB/raw/W7X/QSR_Limiter_NIR/"
              "AEF10_A_Port_DATASTREAM")
    t1, t2 = "2016-03-10 15:00:29.9", "2016-03-10 15:00:30"
    t, d = images.get_image_png_multiple(signal, t1, t2)
    assert t.shape[0] == 5
    assert d.shape == (5, 288, 720)
    stored = np.load("tests/data/image_png_mult_1.npz")
    ts = stored["t"]
    ds = stored["d"]
    assert (t == ts).min()
    assert (d == ds).min()
