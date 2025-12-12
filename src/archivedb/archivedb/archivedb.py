# -*- coding: utf-8 -*-
"""
    A set of functions to work with W7-X Archivedb via web-api.

    Simple function to read from and to write to the database are available.
    This includes reading/writing arrays, images. Also reading of
    performed programs is available.
"""
from builtins import int
from . import cache
from . import url
from . import utils
from .programs import *
from .versions import *
from .timing import *
from .parlogs import *
from .signals import *
from .images import *
from .version import __version__
from .url import _mdsplus_url

__author__ = u"boz dsk"
__maintainer__ = u"boz"
__email__ = "boz at ipp"


# TODO
# 4. reduce interface to single get_signal, do the interval/etc. logic
# internally


def return_library_version():
    """
        Return current library version.
    """
    return __version__


def print_info():
    """
        Print version, author, etc.
    """
    print(__doc__)
    print("Version: {}".format(__version__))
    print("Author: {}".format(__author__))
    print("Cache path: {}".format(cache._cache_path))
    print("DB URL: {}".format(url._db_url))
    print("MDSPlus: {}".format(_mdsplus_url))
