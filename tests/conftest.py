import pytest


@pytest.fixture(scope="session")
def data():
    from traffic.data import load_metrla
    return load_metrla()


@pytest.fixture(scope="session")
def fb(data):
    from traffic.features import FeatureBuilder
    return FeatureBuilder(data)
