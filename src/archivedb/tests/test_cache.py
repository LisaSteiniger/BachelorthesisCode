from archivedb import cache
import os
import numpy as np
import time
import pytest
import shutil

def test_shoud_read_cache(unblock_cache):
    # global flags
    assert not cache._should_read_cache({"forceCacheUpdate": True,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert not cache._should_read_cache({"forceCacheUpdate": True,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": True,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_FS)
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": True,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_MEM)
    cache._block_cache = True
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    cache._block_cache = False

    # cache type selector
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_FS)
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_FS)
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_MEM)
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_MEM)
    # safe context
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_FS)
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_FS)
    assert not cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_MEM)
    assert cache._should_read_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_MEM)

def test_shoud_write_cache(unblock_cache):
    # global flags
    assert cache._should_write_cache({"forceCacheUpdate": True,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert cache._should_write_cache({"forceCacheUpdate": True,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": True,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": True,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    cache._block_cache = True
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_FS)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        True, cache.CACHE_MEM)
    cache._block_cache = False
    # cache type selector
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_FS)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_FS)
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_MEM)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        True, cache.CACHE_MEM)
    # # safe context
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_FS)
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": False,
                                         "useFSCache": True,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        False, cache.CACHE_FS)
    assert not cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": True,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": False},
                                        False, cache.CACHE_MEM)
    assert cache._should_write_cache({"forceCacheUpdate": False,
                                         "useMemCache": True,
                                         "useFSCache": False,
                                         "readUnsafeCache": False,
                                         "workOnlyWithCache": False,
                                         "writeUnsafeCache": True},
                                        False, cache.CACHE_MEM)

def test_compose_cache_filename():
    res = cache._compose_cache_filename_for_signals(["aa/bb/cc",
                                                     "20181016.037",
                                                     1,2, "key1", "v1"],
                                                    prefix="ppref",
                                                    suffix=".suf",
                                                    path = "/any/path/")
    out = "/any/path/aa/bb/cc/ppref_20181016.037_1_2_key1_v1.suf"
    assert res == out
    res = cache._compose_cache_filename_for_proginfo([
                                                     "20181016.037",
                                                     1,2, "key1", "v1"],
                                                    prefix="ppref",
                                                    suffix=".suf",
                                                    path = "/any/path/")
    out = "/any/path//ppref_20181016.037_1_2_key1_v1.suf"
    assert res == out
    url_ = "http://archive.ipp.mpg.de/signal/path/_signal.json"
    res = cache._compose_cache_filename_for_raw([url_, "protocol",
                                                 "json"],
                                                path="/any/path/")
    assert res == "/any/path//" + url_[7:] + "_protocol_json"
    # the same with a query string, which should be transformed into hash
    url_ = ("http://archive.ipp.mpg.de/signal/path/_signal.json?from="
            "112212&to=144347&channels=1,2,3,4,5,6,7,8")
    res = cache._compose_cache_filename_for_raw([url_, "protocol",
                                                 "json"],
                                                path="/any/path/")
    parts = url_.split("?")
    assert res == ("/any/path//" + parts[0][7:] + "?" +
                   "1e06d884b882f6da28da626aa2639068c1368a9bf6a379e4b"
                   "e305819c51564462caf5ac96552dc8911f10df456ab9d57b6fa4"
                   "a2abc1f32ad66032cc1aa0caad6" + "_protocol_json")
    # but if several ? are present -> striaght string, bcs. Unexpected format
    url_ = ("http://archive.ipp.mpg.de/signal/path/_signal.json?from="
            "112212&to=144347&channels=1,2,3,4,5,6,7,8?2?3")
    res = cache._compose_cache_filename_for_raw([url_, "protocol",
                                                 "json"],
                                                path="/any/path/")
    assert res == "/any/path//" + url_[7:] + "_protocol_json"

    # test list
    res = cache._compose_cache_filename_for_signals(["aa/bb/cc",
                                                     [1,2,3]],
                                                    path="")
    out = "aa/bb/cc/[1,2,3]"
    assert res == out

    # test np.array
    res = cache._compose_cache_filename_for_signals(["aa/bb/cc",
                                                     np.array([1,2,3])],
                                                    path="")
    out = "aa/bb/cc/[1,2,3]"
    assert res == out


def test_check_cache_directory(tmp_path):
    # todo: fix for wins
    path = tmp_path.as_posix()
    cache._check_cache_directory(path)
    assert os.path.exists(path)
    cache._check_cache_directory(path)
    assert os.path.exists(path)

def test_open_read_write(tmp_path):
    # todo: fix for wins
    path = tmp_path.as_posix() + "test.txt"
    with cache._open_file_for_writing(path, "t") as f:
        f.write("test string")
    with cache._open_file_for_reading(path, "t") as f:
        res = f.read()
    assert res == "test string"

def test_put_get_signal(tmp_path):
    # todo: fix for wins
    path = tmp_path.as_posix() + "test.npz"
    N = 10
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    cache._put_signal_to_fscache(path, [[t,s], [t0]], 0)
    ret = cache._get_signal_from_fscache(path, None)
    assert ret is not None
    assert (ret[0][0] == t).min()
    assert (ret[0][1] == s).min()
    assert ret[1][-1] == t0
    s2 = np.random.uniform(size=N*2).reshape(2, N)
    s3 = np.random.uniform(size=N*4).reshape(2, 2, N)
    cache._put_signal_to_fscache(path, [[t, s2, s3, s], [t0]], 0)
    ret = cache._get_signal_from_fscache(path, None)
    assert ret is not None
    assert(len(ret[0]) == 4)
    assert (ret[0][0] == t).min()
    assert (ret[0][1] == s2).min()
    assert (ret[0][2] == s3).min()
    assert (ret[0][3] == s).min()
    assert ret[1][-1] == t0
    # check non-tuple, single array
    cache._put_signal_to_fscache(path, [ s3, [t0]], 0)
    ret = cache._get_signal_from_fscache(path, None)
    assert ret is not None
    assert (ret[0] == s3).min()
    assert ret[1][-1] == t0


def test_put_get_intervals(tmp_path):
    # todo: fix for wins
    path = tmp_path.as_posix() + "test2.npz"
    N = 10
    t0 = int(time.time())*1000000000
    s = np.random.uniform(size=N*2).reshape(2, N)
    cache._put_intervals_to_fscache(path, [s, [t0]], 0)
    ret = cache._get_intervals_from_fscache(path, None)
    assert ret is not None
    assert (ret[0] == s).min()
    assert ret[1][-1] == t0

def test_put_get_json(tmp_path):
    # todo: fix for wins
    path = tmp_path.as_posix() + "test3.json"
    t0 = int(time.time())
    s = {"key1": "v1", "key2": 12123, "key3": [3,4,5]}
    cache._put_json_to_fscache(path, [s, [t0]], 0)
    ret = cache._get_json_from_fscache(path, None)
    assert ret is not None
    assert ret[0] == s
    assert ret[1][-1] == t0

def test_put_get_memcache():
    storage = cache.OrderedDict()
    lock = cache.Lock()
    # simple storage for different types
    ## signal type
    N = 10
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    cache._put_to_memcache(storage, lock, "key1", [[t,s],[t0,]], 100)
    assert "key1" in storage
    ret = cache._get_from_memcache(storage, lock, "key1", None)
    assert ret is not None
    assert (ret[0][0] == t).min()
    assert (ret[0][1] == s).min()
    assert ret[1][-1] == t0
    ## intervals type
    N = 10
    t0 = int(time.time())*1000000000
    s = np.random.uniform(size=N*2).reshape(2, N)
    cache._put_to_memcache(storage, lock, "key2", [s, [t0,]], 100)
    assert "key1" in storage
    assert "key2" in storage
    ret = cache._get_from_memcache(storage, lock, "key2" , None)
    assert ret is not None
    assert (ret[0] == s).min()
    assert ret[1][-1] == t0
    ## dict
    t0 = int(time.time())
    s = {"key1": "v1", "key2": 12123, "key3": [3,4,5]}
    cache._put_to_memcache(storage, lock, "key3", [s, [t0,]], 100)
    assert "key1" in storage
    assert "key2" in storage
    assert "key3" in storage
    ret = cache._get_from_memcache(storage, lock, "key3", None)
    assert ret is not None
    assert ret[0] == s
    assert ret[1][-1] == t0
    # cache rotation
    ret = cache._get_from_memcache(storage, lock, "key1" , None)
    assert "key1" in storage
    assert "key2" in storage
    assert "key3" in storage
    assert list(storage.keys()) == ["key2", "key3", "key1"]
    cache._put_to_memcache(storage, lock, "key4", [s, [t0]], 3)
    assert "key1" in storage
    assert "key3" in storage
    assert "key4" in storage
    assert "key2" not in storage
    assert len(storage) == 3
    assert list(storage.keys()) == ["key3", "key1", "key4"]
    ret = cache._get_from_memcache(storage, lock, "key1" , None)
    cache._put_to_memcache(storage, lock, "key5", [s, [t0]], 2)
    assert "key1" in storage
    assert "key2" not in storage
    assert "key3" not in storage
    assert "key4" not in storage
    assert "key5" in storage
    assert len(storage) == 2
    assert list(storage.keys()) == ["key1", "key5",]

def is_safe(*args, **kwargs):
    if "safe" not in kwargs:
        return True
    return kwargs["safe"]

@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.EXTERNAL,
                           safe_context=is_safe,
                           hash_keywords=["hashable"],
                           cache_path="",
                           settings={"useMemCache": True,
                                          "maxCacheLength": 3})
def dummy1(signal, shot, hashable=None, useCache=True,
          cacheSettings=None, doRaise=False, safe=True):
    if doRaise:
        raise RuntimeError("shouldn't happen")
    return np.random.uniform()

def test_memcache(unblock_cache):
    assert hasattr(dummy1, "__memcache_storage__")
    assert hasattr(dummy1, "__memcache_lock__")
    assert hasattr(dummy1, "__memcache_settings__")
    assert len(dummy1.__memcache_storage__) == 0
    assert len(dummy1.__memcache_settings__) == 2
    res1 = dummy1("call1", 10, hashable="boom")
    assert len(dummy1.__memcache_storage__) == 1
    key1 = "call1/10_hashable_boom"
    assert key1 in dummy1.__memcache_storage__
    assert dummy1.__memcache_storage__[key1][0] == res1
    # will raise an exception if the inner function is called
    assert dummy1("call1", 10, hashable="boom",
                  doRaise=True) == res1
    res2 = dummy1("call2", 20, hashable="boom")
    assert len(dummy1.__memcache_storage__) == 2
    key2 = "call2/20_hashable_boom"
    assert key2 in dummy1.__memcache_storage__
    res3 = dummy1("call3", 30)
    assert len(dummy1.__memcache_storage__) == 3
    key3 = "call3/30_hashable_None"
    assert key3 in dummy1.__memcache_storage__
    # non safe without write -> no cache update
    res4 = dummy1("call4", 40, cacheSettings={"writeUnsafeCache":False},
                  safe=False)
    key4 = "call4/40_hashable_None"
    assert len(dummy1.__memcache_storage__) == 3
    assert key1 in dummy1.__memcache_storage__
    assert key2 in dummy1.__memcache_storage__
    assert key3 in dummy1.__memcache_storage__
    assert key4 not in dummy1.__memcache_storage__
    # access reoder 
    assert dummy1("call1", 10, hashable="boom",
                  doRaise=True) == res1
    assert dummy1("call2", 20, hashable="boom",
                  doRaise=True) == res2
    assert dummy1("call1", 10, hashable="boom",
                  doRaise=True) == res1
    assert (list(dummy1.__memcache_storage__.keys()) ==
            [key3, key2, key1])
    # unsafe with write and drop oldest
    res4 = dummy1("call4", 40, cacheSettings={"writeUnsafeCache":True},
                  safe=False)
    assert key1 in dummy1.__memcache_storage__
    assert key2 in dummy1.__memcache_storage__
    assert key3 not in dummy1.__memcache_storage__
    assert key4 in dummy1.__memcache_storage__
    assert dummy1("call1", 10, hashable="boom",
                  doRaise=True) == res1
    assert dummy1("call2", 20, hashable="boom",
                  doRaise=True) == res2
    assert dummy1("call4", 40, doRaise=True) == res4
    with pytest.raises(RuntimeError):
        assert dummy1("call3", 30, doRaise=True) != res3

@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.EXTERNAL,
                           safe_context=is_safe,
                           hash_keywords=["hashable"],
                           cache_path="tests/cache", #????
                           name_prefix="dummy2",
                           name_suffix=".npz",
                           put_to_cache=cache._put_signal_to_fscache,
                           get_from_cache=cache._get_signal_from_fscache)
def dummy2(signal, shot, hashable=None, useCache=True,
          cacheSettings=None, doRaise=False, safe=True,
           N = 100):
    if doRaise:
        raise RuntimeError("shouldn't happen")
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    return t, s # emulate signal return

def test_fscache(unblock_cache):
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")
    assert hasattr(dummy2, "__fscache_settings__")
    assert len(dummy2.__fscache_settings__) == 0
    res1 = dummy2("call1", 10, hashable="boom")
    key1 = "tests/cache/call1/dummy2_10_hashable_boom.npz"
    assert os.path.exists(key1)
    res11 = dummy2("call1", 10, hashable="boom", doRaise=True)
    assert (res1[0] == res11[0]).min()
    assert (res1[1] == res11[1]).min()
    res2 = dummy2("call2", 20, hashable=None)
    key2 = "tests/cache/call2/dummy2_20_hashable_None.npz"
    assert os.path.exists(key1)
    assert os.path.exists(key2)
    res22 = dummy2("call2", 20, hashable=None, doRaise=True)
    assert (res2[0] == res22[0]).min()
    assert (res2[1] == res22[1]).min()
    res11 = dummy2("call1", 10, hashable="boom", doRaise=True)
    assert (res1[0] == res11[0]).min()
    assert (res1[1] == res11[1]).min()
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")


def test_is_not_too_old():
    t = int(time.time())
    res = [[1, 2], ["addon", "path", t-100000]]
    settings = {"cacheExpireTime" : 1000,
                "workOnlyWithCache": False}
    assert cache._is_not_too_old(res, True, settings, None)
    assert not cache._is_not_too_old(res, False, settings, None)
    res = [[1, 2], ["addon", "path", t-100]]
    assert cache._is_not_too_old(res, False, settings, None)
    res = [[1, 2], ["addon", "path", t-100000]]
    settings = {"cacheExpireTime" : 0,
                "workOnlyWithCache": True}
    assert cache._is_not_too_old(res, True, settings, None)


def test_is_valid_aliasedsignal():
    t = int(time.time())
    res = [[1, 2], ["addon", "path", t-100000]]
    settings = {"cacheExpireTime" : 1000,
                "workOnlyWithCache": False}
    assert cache._is_valid_aliasedsignal(res, True, settings, None,
                                         path="path")
    assert not cache._is_valid_aliasedsignal(res, False, settings, None,
                                             path="path")
    assert not cache._is_valid_aliasedsignal(res, True, settings, None,
                                             path="path2")
    res = [[1, 2], [t-100000]]
    assert not cache._is_valid_aliasedsignal(res, True, settings, None,
                                             path="path2")


@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.ALIASEDSIGNAL,
                           safe_context=lambda *args, **kwargs: True,
                           cache_path="tests/cache", #????
                           name_prefix="dummy3",
                           name_suffix=".npz")
def dummy3(signal, shot, useCache=True,
           cacheSettings=None, doRaise=False,
           path="defaultpath", returnSignalPath=False,
           N = 100):
    if doRaise:
        raise RuntimeError("shouldn't happen")
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    if returnSignalPath:
        return (t,s,), path
    return t, s # emulate signal return


def test_fscache_for_aliasedsignal(unblock_cache):
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")
    res1 = dummy3("call1", 10)
    key1 = "tests/cache/call1/dummy3_10.npz"
    assert os.path.exists(key1)
    res2, path = dummy3("call1", 10, useCache=False,
                        returnSignalPath=True)
    assert path == "defaultpath"
    assert (res1[0] == res2[0]).min()
    assert not (res1[1] == res2[1]).min()
    res3, path = dummy3("call1", 10, doRaise=True,
                        path="defaultpath",
                        returnSignalPath=True)
    assert path == "defaultpath"
    assert (res3[0] == res1[0]).min()
    assert (res3[1] == res1[1]).min()
    res4 = dummy3("call1", 10, doRaise=True,
                  path="defaultpath",
                  returnSignalPath=False)
    assert (res4[0] == res1[0]).min()
    assert (res4[1] == res1[1]).min()
    with pytest.raises(RuntimeError):
        res5, path = dummy3("call1", 10, doRaise=True,
                        path="defaultpath2")
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")


@cache.cache_this_function(cache_type=cache.CACHE_MEM,
                           data_type=cache.ALIASEDSIGNAL,
                           safe_context=lambda *args, **kwargs: True)
def dummy4(signal, shot, useCache=True,
           cacheSettings=None, doRaise=False,
           path="defaultpath", returnSignalPath=False,
           N = 100):
    if doRaise:
        raise RuntimeError("shouldn't happen")
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    if returnSignalPath:
        return (t,s,), path
    return t, s # emulate signal return


def test_memcache_for_aliasedsignal(unblock_cache):
    res1 = dummy4("call1", 10,
                   cacheSettings={"useMemCache": True,
                                  "maxCacheLength":10},)
    res2, path = dummy4("call1", 10,
                        cacheSettings={"useMemCache": False},
                        returnSignalPath=True)
    assert path == "defaultpath"
    assert (res1[0] == res2[0]).min()
    assert not (res1[1] == res2[1]).min()
    res3, path = dummy4("call1", 10, doRaise=True,
                        cacheSettings={"useMemCache": True,
                                       "maxCacheLength":10},
                        path="defaultpath",
                        returnSignalPath=True)
    assert path == "defaultpath"
    assert (res3[0] == res1[0]).min()
    assert (res3[1] == res1[1]).min()
    res4 = dummy4("call1", 10, doRaise=True,
                  cacheSettings={"useMemCache": True,
                                 "maxCacheLength":10},
                  path="defaultpath",
                  returnSignalPath=False)
    assert (res4[0] == res1[0]).min()
    assert (res4[1] == res1[1]).min()
    with pytest.raises(RuntimeError):
        res5, path = dummy4("call1", 10, doRaise=True,
                      cacheSettings={"useMemCache": True,
                                     "maxCacheLength":10},
                        path="defaultpath2")


def test_isaliasedsignal_safe():
    assert not cache._is_aliasedsignal_safe()
    assert not cache._is_aliasedsignal_safe(path="1/views/fdfd/V3/0")
    assert not cache._is_aliasedsignal_safe(path="1/Test/fdfd/XV3/0")
    assert cache._is_aliasedsignal_safe(path="1/Test/fdfd/V3/0")
    assert cache._is_aliasedsignal_safe(path="1/Test/fdfd/V3")



@cache.cache_this_function(cache_type=cache.CACHE_FS,
                           data_type=cache.EXTERNAL,
                           safe_context=is_safe,
                           hash_keywords=["hashable"],
                           cache_path="tests/cache", #????
                           name_prefix="dummy5",
                           name_suffix=".npz",
                           put_to_cache=cache._put_signal_to_fscache,
                           get_from_cache=cache._get_signal_from_fscache,
                           save_input_parameters=True)
def dummy5(signal, shot, hashable=None, useCache=True,
          cacheSettings=None, doRaise=False, safe=True,
           N = 100):
    if doRaise:
        raise RuntimeError("shouldn't happen")
    t0 = int(time.time())*1000000000
    t = np.arange(N) + t0
    s = np.random.uniform(size=N*2).reshape(2, N)
    return t, s # emulate signal return

def test_fscache_save_input(unblock_cache):
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")
    res1 = dummy5("call1", 10, hashable="boom")
    key1 = "tests/cache/call1/dummy5_10_hashable_boom.npz"
    assert os.path.exists(key1)
    res11 = dummy5("call1", 10, hashable="boom", doRaise=True)
    assert (res1[0] == res11[0]).min()
    assert (res1[1] == res11[1]).min()
    res2 = dummy5("call2", 20, hashable=None)
    key2 = "tests/cache/call2/dummy5_20_hashable_None.npz"
    assert os.path.exists(key1)
    assert os.path.exists(key2)
    x = np.random.normal(size=(2, 3))
    res3 = dummy5("call3", 30, hashable=x)
    key3 = ("tests/cache/call3/dummy5_30_hashable_" +
            cache._tostring(x) + ".npz")
    assert os.path.exists(key1)
    assert os.path.exists(key2)
    assert os.path.exists(key3)
    xx = np.load(key3)["info"][-3]
    yy = "_".join([cache._tostring(x) for x in ["call3", "30",
                                                "hashable", x]])
    assert xx == yy
    if os.path.exists("tests/cache"):
        shutil.rmtree("tests/cache")


def test_get_hash():
    tmp = np.linspace(0,10,30).reshape(10, 3)
    res = cache._get_hash([tmp, "w7x_ref_3", [1, "a", 3.14], 10.5])
    assert res == ("f32f6fe7b12eded105c47e24ee06e5bdfb838a6615b5c0e03829ee4"
                   "e234a9257c35fd16c87f4522115b87b38144c7f47c05d0d41"
                   "40a20571e34a255627c618c4")
    # test slicing the second array dimension, bcs. this produces
    # a non-contiguous array that fails with hashlib, unless handled
    # properly in the function
    x = np.linspace(0, 10, 100).reshape(50, 2)
    res = cache._get_hash([tmp, "w7x_ref_3", x[:, 1], 10.5])
    assert res == ("e80f340357bde0952dd3efe56d62e91e0238280984612ccdead"
                   "dd466225cc8c4b76b99a7087408fbafd8f91d9c267181a806ce3b"
                   "92bd89ebb4d4bfc7e800af2c")


def test_compose_hash_cache_filename():
    tmp = np.linspace(0,10,30).reshape(10, 3)
    res = cache._compose_hash_cache_filename([tmp,
                                              "w7x_ref_3",
                                              [1, "a", 3.14], 10.5],
                                               prefix="ppref",
                                               suffix=".suf",
                                               path = "/any/path/")
    out = ("/any/path//ppref_f32f6fe7b12eded105c47e24ee06e5bd"
           "fb838a6615b5c0e03829ee4e234a9257c35fd16c87f4522115b"
           "87b38144c7f47c05d0d4140a20571e34a255627c618c4.suf")
    assert res == out
    res = cache._compose_hash_cache_filename([tmp,
                                              "w7x_ref_3",
                                              [1, "a", 3.145], 10.5],
                                               prefix="",
                                               suffix="",
                                               path = "")
    out = ("17bbe38bea2987bd4fd085632c431a39a3a28d"
           "ce3b1693599f8f280b3c3264747489f4360a548"
           "04209ccbf2b3aca1293412c27f61b0ac342576fd8ea0aa2a403")
    assert res == out
    # test slicing the second array dimension, bcs. this produces
    # a non-contiguous array that fails with hashlib, unless handled
    # properly in the function
    x = np.linspace(0, 10, 100).reshape(50, 2)
    res = cache._compose_hash_cache_filename([tmp,
                                              "w7x_ref_3",
                                              x[:, 1], 10.5],
                                               prefix="",
                                               suffix="",
                                               path = "")
    out = ("e80f340357bde0952dd3efe56d62e91e0238280984612c"
            "cdeaddd466225cc8c4b76b99a7087408fbafd8f91d9c267"
            "181a806ce3b92bd89ebb4d4bfc7e800af2c")
    assert res == out


def test_compose_hash_cache_filename_with_signame():
    tmp = np.linspace(0,10,30).reshape(10, 3)
    res = cache._compose_hash_cache_filename_with_signame(
        ["Test/raw/W7X/Stream", "w7x_ref_3", [1, "a", 3.14], 10.5],
        prefix="ppref", suffix=".suf", path = "/any/path/")
    out = ("/any/path/Test/raw/W7X/Stream/ppref_545c6e2925427842f309d3009"
           "5b6d3a5d3000c4d4eec8e526fccbfe5b2411aa80881"
           "2252a99ef7bb0ecb7000f7ea94f3a0d3aeb1e3f1c985a2c6fdcd35344961.suf")
    assert res == out


def test_is_valid_input_string_in_cache():
    t = int(time.time())
    hash_args = [1.0/3.0, "key1", "rear", np.array([1.3, 2.3])]
    h = "_".join([cache._tostring(x) for x in hash_args])
    res = [[1, 2], ["addon", h, "", t-100000]]
    settings = {"cacheExpireTime" : 1000,
                "workOnlyWithCache": False}
    assert cache._is_valid_input_string_in_cache(res, True, settings,
                                                 hash_args)
    assert not cache._is_valid_input_string_in_cache(res, False, settings,
                                                     hash_args)
    assert not cache._is_valid_input_string_in_cache(res, True, settings,
                                                     hash_args + [1])
    res = [[1, 2], [t-100000]]
    assert not cache._is_valid_input_string_in_cache(res, True, settings,
                                                     hash_args)

    res = [[1, 2], ["addon", h, "", t-100000]]
    hash_args = [1.0/3.0, "key1", "rear", np.array([1.3, 2.3])]
    h = "_".join([cache._tostring(x) for x in hash_args])
    assert cache._is_valid_input_string_in_cache(res, True, settings,
                                                 hash_args)
    hash_args = [1.0/3.0 + 1e-16, "key1", "rear", np.array([1.3, 2.3])]
    assert not cache._is_valid_input_string_in_cache(res, True, settings,
                                                 hash_args)
