"""Fixtures for the store template-method acceptance tests.

No real Redis server is required: when ``DIFFSYNC_TEST_REDIS_URL`` is not set,
``fakeredis`` provides the Redis client instead.
"""

from typing import ClassVar, List, Tuple

import pytest

from diffsync import DiffSyncModel


class Site(DiffSyncModel):
    """Minimal site model."""

    _modelname = "site"
    _identifiers = ("name",)
    _children = {"device": "devices"}

    name: str
    devices: List = []


class Device(DiffSyncModel):
    """Minimal device model."""

    _modelname = "device"
    _identifiers = ("name",)
    _attributes: ClassVar[Tuple[str, ...]] = ("role",)

    name: str
    role: str = "default"


@pytest.fixture
def make_site():
    """Factory for Site instances."""

    def _make(name="site1", **kwargs):
        return Site(name=name, **kwargs)

    return _make


@pytest.fixture
def make_device():
    """Factory for Device instances."""

    def _make(name="device1", role="default", **kwargs):
        return Device(name=name, role=role, **kwargs)

    return _make


@pytest.fixture
def redis_client():
    """Provide a Redis-compatible client.

    Uses ``fakeredis`` unless a real server URL is provided through the
    ``DIFFSYNC_TEST_REDIS_URL`` environment variable.
    """
    import os

    url = os.environ.get("DIFFSYNC_TEST_REDIS_URL")
    if url:
        from redis import Redis

        return Redis.from_url(url)
    import fakeredis

    return fakeredis.FakeRedis()


@pytest.fixture(autouse=True)
def _isolate_store_registry():
    """Keep the global store registry empty around each test."""
    from diffsync import store as store_module

    store_module._clear_registered_stores()
    yield
    store_module._clear_registered_stores()
