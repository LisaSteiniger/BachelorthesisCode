import pytest

@pytest.fixture
def block_cache():
    from archivedb import cache
    cache._block_cache = True

@pytest.fixture
def unblock_cache():
    from archivedb import cache
    cache._block_cache = False
