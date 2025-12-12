from archivedb import aliases
from archivedb import programs
import logging

def test_resolve_alias(block_cache):
    logging.getLogger(__name__).warning("Attention, aliases "
                                        "can change with time. "
                                        "Therefore, this test may "
                                        "fail.")
    signal = ("Test/views/Minerva/PlasmaCurrent/"
              "RogowskiCoil.Continuous/Iplasma/signal")
    shot = "20181004.020"
    res = ("Test/raw/Minerva1/"
           "Minerva.Magnetics15.Iplasma/Iplasma_QXR11CE001x_DATASTREAM/"
           "V2/0/Iplasma_for_continous_Rogowski_QXR11CE001x")
    assert aliases.resolve_alias_for_program(signal, shot,
                                             timeout=10) == res
    t1, t2 = programs.get_program_from_to(shot, timeout=10)
    assert aliases.resolve_alias(signal, t1, t2, timeout=10) == res

@aliases.convert_alias_to_signal
def dummy1(signal, shot, t1, t2, key1=None, key2=None,
           timeout=1, useCache=True, cacheSettings=None):
    return (signal, shot, t1, t2, key1, key2,
            timeout, useCache, cacheSettings)

@aliases.convert_alias_to_signal
def dummy2(signal, shot, t1, t2, key1=None, key2=None,
           timeout=10):
    return (signal, shot, t1, t2, key1, key2)

def test_alias_decorator(block_cache):
    alias = ("Test/views/Minerva/PlasmaCurrent/"
              "RogowskiCoil.Continuous/Iplasma/signal")
    shot = "20181004.020"
    signal = ("Test/raw/Minerva1/"
           "Minerva.Magnetics15.Iplasma/Iplasma_QXR11CE001x_DATASTREAM/"
           "V2/0/Iplasma_for_continous_Rogowski_QXR11CE001x")
    res1 = (signal, shot, 2, 3, "k1k", "k2k", 10, False,
            {"dodo": "nono"})
    assert dummy1(alias, shot, 2,3, key1="k1k",
                  key2="k2k", timeout=10, useCache=False,
                  cacheSettings=res1[-1]) == res1
    res2 = (signal, shot, 2, 3, "k1k", "k2k")
    assert dummy2(alias, shot, 2,3, key1="k1k",
                  key2="k2k", timeout=10) == res2

