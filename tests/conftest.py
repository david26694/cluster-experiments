import datetime
from collections import namedtuple

import pytest

Duration = namedtuple("Duration", ["current", "last"])


@pytest.fixture(scope="session")
def duration_cache(request):
    """We can't use `cache` fixture because it has function scope. However the `cache`
    fixture simply returns `request.config.cache`, which is available in any scope."""
    key = "duration/testdurations"
    d = Duration({}, request.config.cache.get(key, {}))
    yield d
    request.config.cache.set(key, d.current)


@pytest.fixture(autouse=True)
def check_duration(request, duration_cache):
    d = duration_cache
    nodeid = request.node.nodeid
    start_time = datetime.datetime.now()
    yield
    duration = (datetime.datetime.now() - start_time).total_seconds()
    d.current[nodeid] = duration


def by_duration(item):
    return item.config.cache.get("duration/testdurations", {}).get(item.nodeid, 0)


def pytest_addoption(parser):
    parser.addoption("--slow-last", action="store_true", default=False)


# Runs tests marked with slow last
def pytest_collection_modifyitems(items, config):
    if config.getoption("--slow-last"):
        items.sort(key=by_duration, reverse=False)
