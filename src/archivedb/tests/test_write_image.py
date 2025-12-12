from archivedb import images
from archivedb import parlogs
import numpy as np
import time
import PIL

def test_write_image(block_cache):
    im = PIL.Image.open("tests/data/w7x.jpg")
    width, height = im.size
    image = np.array(im.getdata()).reshape(height, width,
                                           3).sum(axis=-1)
    d =  {"projectName":"test data", "cameraType":"test data",
          "height" : 287, "width" : 380, "bitDepth" : 10,
          "dataBoxSize" : 1, "datatype" : "short", "unsigned" : 1}
    camname = "bzzz"
    signal = "Test/raw/W7XAnalysis/HDF5_import_test/"
    t0 = int(round(time.time() * 1000)) * 1000000
    assert images.write_image_to_database(signal, t0, image, d,
                                          camname=camname)
    d2 = parlogs.get_parameters_box(signal + camname + "_PARLOG",
                                    t0, -1)
    assert d2["dimensions"] == [t0, t0]
    for k in d: # d2 has added chanDesc
        assert d2["values"][0][k] == d[k]

    t3, d3 = images.get_image(signal + camname + "_DATASTREAM",
                              t0, -1)
    assert (t3 == t0).min()
    d3 = d3.squeeze()
    assert d3.shape == image.shape
    assert (d3 == image).min()
