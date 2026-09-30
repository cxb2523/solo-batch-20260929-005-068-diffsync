"""Acceptance tests for RedisStore using fakeredis as the fallback."""

import copy
import pickle

import pytest

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound, ObjectStoreException
from diffsync.store.redis import REDIS_DIFFSYNC_ROOT_LABEL, RedisStore


def test_unavailable_redis_raises_object_store_exception():
    # No host/url/client: the constructor must fail eagerly, as historically.
    with pytest.raises(ObjectStoreException):
        RedisStore(name="mystore", store_id="123")


def test_url_and_host_are_mutually_exclusive(redis_client):
    with pytest.raises(ValueError):
        RedisStore(store_id="123", url="redis://localhost", host="localhost", client=redis_client)


def test_init_str_and_label(redis_client):
    store = RedisStore(name="mystore", store_id="123", client=redis_client)
    assert str(store) == "mystore (123)"
    assert store._store_label == f"{REDIS_DIFFSYNC_ROOT_LABEL}:123"


def test_crud_and_key_layout(redis_client, make_site):
    store = RedisStore(name="mystore", store_id="123", client=redis_client)
    site = make_site()
    store.add(obj=site)
    assert store.count() == 1
    # Exact historical physical key layout must be preserved.
    assert redis_client.exists(b"diffsync:123:site:site1")
    assert store.get(model=site.__class__, identifier="site1") == site
    assert store.get_all(model=site.__class__) == [site]
    assert store.get_by_uids(uids=["site1"], model=site.__class__) == [site]
    store.remove(obj=site)
    assert store.count() == 0


def test_legacy_payload_and_key_layout_read_back(redis_client, make_site):
    """Data written by the pre-refactor RedisStore reads back unchanged."""
    store = RedisStore(store_id="legacy", client=redis_client)
    site = make_site(name="ancient")
    obj_copy = copy.copy(site)
    obj_copy.adapter = None
    redis_client.set("diffsync:legacy:site:ancient", pickle.dumps(obj_copy))
    got = store.get(model=site.__class__, identifier="ancient")
    assert got == site
    assert store.count() == 1
    assert store.get_all_model_names() == {"site"}


def test_new_writes_are_byte_compatible_with_legacy_codec(redis_client, make_site):
    store = RedisStore(store_id="compat", client=redis_client)
    site = make_site(name="new")
    store.add(obj=site)
    raw = redis_client.get("diffsync:compat:site:new")
    # The payload is a plain pickle with no version marker or wrapper.
    decoded = pickle.loads(raw)  # noqa: S301
    assert decoded == site
    assert decoded.adapter is None
    assert raw[:1] == b"\x80"  # pickle protocol opcode


def test_add_twice_is_noop_and_conflict_raises(redis_client, make_site, make_device):
    store = RedisStore(store_id="123", client=redis_client)
    site = make_site()
    store.add(obj=site)
    store.add(obj=site)
    assert store.count() == 1
    # Same namespace+uid as an existing device but different content.
    store.add(obj=make_device(name="d1", role="leaf"))
    with pytest.raises(ObjectAlreadyExists):
        store.add(obj=make_device(name="d1", role="spine"))
    assert store.count() == 2


def test_remove_missing_raises(redis_client):
    store = RedisStore(store_id="123", client=redis_client)
    with pytest.raises(ObjectNotFound):
        store.remove_item("site", "ghost")


def test_keys_prefix_isolation_between_store_ids(redis_client, make_site):
    store_a = RedisStore(store_id="aaa", client=redis_client)
    store_b = RedisStore(store_id="bbb", client=redis_client)
    store_a.add(obj=make_site())
    assert store_a.count() == 1
    assert store_b.count() == 0
    assert store_a.keys() == [("site", "site1")]
    assert store_b.keys() == []


def test_invalid_namespace_and_key_do_not_touch_redis(redis_client):
    store = RedisStore(store_id="123", client=redis_client)
    with pytest.raises(ValueError):
        store.set("k", "v", namespace="bad:ns")
    with pytest.raises(ValueError):
        store.set("bad:k", "v", namespace="ns")
    assert list(redis_client.scan_iter("diffsync:123:*")) == []


def test_context_manager_closes_client():
    import fakeredis

    with RedisStore(store_id="cm", client=fakeredis.FakeRedis()) as store:
        store.set("k", "v", namespace="ns")
        assert store.get("k", namespace="ns") == "v"
    assert store._connection is None


def test_backwards_compatible_helpers(redis_client, make_site):
    store = RedisStore(store_id="123", client=redis_client)
    store.add(obj=make_site())
    obj = store._get_object_from_redis_key("diffsync:123:site:site1")
    assert obj == make_site()
    assert store._get_key_for_object("site", "site1") == "diffsync:123:site:site1"


def test_generic_kv_api(redis_client):
    store = RedisStore(store_id="kv", client=redis_client)
    store.set("k", {"x": 1}, namespace="ns")
    assert store.exists("k", namespace="ns")
    assert store.get("k", namespace="ns") == {"x": 1}
    assert store.keys(namespace="ns") == ["k"]
    store.delete("k", namespace="ns")
    assert not store.exists("k", namespace="ns")
