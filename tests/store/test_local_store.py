"""Acceptance tests for the shelve-backed LocalStore template methods."""

import os

import pytest

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound
from diffsync.store import JSONCodec, PickleCodec
from diffsync.store.local import LocalStore


def test_default_codec_is_pickle():
    store = LocalStore()
    assert store.codec.name == "pickle"
    assert isinstance(store.codec, PickleCodec)
    store.close()


def test_generic_kv_api_roundtrip():
    store = LocalStore(codec=JSONCodec())
    assert store.exists("alpha", namespace="ns") is False
    store.set("alpha", {"v": 1}, namespace="ns")
    assert store.exists("alpha", namespace="ns") is True
    assert store.get("alpha", namespace="ns") == {"v": 1}
    assert store.get("missing", namespace="ns") is None
    assert store.keys(namespace="ns") == ["alpha"]
    store.delete("alpha", namespace="ns")
    assert store.exists("alpha", namespace="ns") is False
    store.close()


def test_keys_lists_namespace_and_key_pairs():
    store = LocalStore(codec=JSONCodec())
    store.set("k1", 1, namespace="a")
    store.set("k2", 2, namespace="b")
    assert store.keys() == [("a", "k1"), ("b", "k2")]
    assert store.keys(namespace="b") == ["k2"]
    store.close()


@pytest.mark.parametrize("namespace", ["", "a:b", "x*", "q?", "z[1]"])
def test_invalid_namespace_raises_before_write(namespace):
    store = LocalStore()
    with pytest.raises(ValueError):
        store.set("k", "v", namespace=namespace)
    assert store.count() == 0
    store.close()


@pytest.mark.parametrize("key", ["", "a:b", "x*", "q?", "z[1]"])
def test_invalid_key_raises_before_write(key):
    store = LocalStore()
    with pytest.raises(ValueError):
        store.set(key, "v", namespace="ns")
    assert store.count() == 0
    store.close()


def test_key_conflict_raises_before_write(make_device):
    store = LocalStore()
    store.add(obj=make_device(name="d1", role="leaf"))
    # Same namespace+uid but different content -> conflict before any second write.
    conflicting = make_device(name="d1", role="spine")
    with pytest.raises(ObjectAlreadyExists):
        store.add(obj=conflicting)
    # Exactly one object remains.
    assert store.count() == 1
    store.close()


def test_add_same_instance_is_noop(make_site):
    site = make_site()
    store = LocalStore()
    store.add(obj=site)
    store.add(obj=site)
    assert store.count() == 1
    assert store.get(model=site.__class__, identifier="site1") is site
    store.close()


def test_remove_missing_raises(make_site):
    store = LocalStore()
    with pytest.raises(ObjectNotFound):
        store.remove_item("site", "ghost")
    store.close()


def test_with_statement_opens_and_closes():
    with LocalStore(codec=JSONCodec()) as store:
        store.set("k", [1, 2], namespace="ns")
        assert store.get("k", namespace="ns") == [1, 2]
    assert store._connection is None


def test_persistence_to_named_path(tmp_path, make_site):
    path = str(tmp_path / "db" / "local")
    site = make_site()
    with LocalStore(path=path) as store:
        store.add(obj=site)
    assert os.path.exists(path + ".db") or any(f.startswith("local") for f in os.listdir(str(tmp_path / "db")))
    with LocalStore(path=path) as reopened:
        assert reopened.count() == 1
        assert reopened.get(model=site.__class__, identifier="site1") == site


def test_identity_semantics_within_session(make_site):
    store = LocalStore()
    site = make_site()
    store.add(obj=site)
    assert store.get(model=site.__class__, identifier="site1") is site
    store.close()


def test_model_names_and_count(make_site, make_device):
    store = LocalStore()
    site = make_site()
    device = make_device()
    store.add(obj=site)
    store.add(obj=device)
    assert store.get_all_model_names() == {"site", "device"}
    assert store.count() == 2
    assert store.count(model="site") == 1
    store.close()


def test_inspection_of_live_session_objects(make_device):
    store = LocalStore()
    store.add(obj=make_device(name="live", role="leaf"))
    report = store.inspect()
    assert report["key_count"] == 1
    assert report["roundtrip_ok"] is True
    store.close()
