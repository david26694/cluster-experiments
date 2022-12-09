from _pytest.mark import Mark

empty_mark = Mark("", [], {})


def by_slow_marker(item):
    return item.get_closest_marker("slow", default=empty_mark)


def pytest_addoption(parser):
    parser.addoption("--slow-last", action="store_true", default=False)


# Runs tests marked with slow last
def pytest_collection_modifyitems(items, config):
    if config.getoption("--slow-last"):
        items.sort(key=by_slow_marker, reverse=False)
