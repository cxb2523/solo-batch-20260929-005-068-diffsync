"""Acceptance tests for the /stores introspection endpoint."""

import pytest
from fastapi.testclient import TestClient

from diffsync.api import app
from diffsync.store import get_registered_stores
from diffsync.store.local import LocalStore
from diffsync.store.redis import RedisStore


@pytest.fixture
def client():
    return TestClient(app)


def test_stores_endpoint_lists_codec_and_counts(client):
    store = LocalStore(name="local-a")
    store.set("one", {"a": 1}, namespace="thing")
    store.set("two", {"b": 2}, namespace="thing")

    response = client.get("/stores")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["total_keys"] == 2
    entry = next(item for item in body["stores"] if item["name"] == "local-a")
    assert entry["codec"] == "pickle"
    assert entry["key_count"] == 2
    assert entry["roundtrip_ok"] is True
    assert entry["roundtrip_failures"] == []


def test_stores_endpoint_readable_text(client):
    store = LocalStore(name="readable")
    store.set("k", "v", namespace="ns")

    response = client.get("/stores?format=text")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "name: readable" in text
    assert "codec: pickle" in text
    assert "keys: 1" in text
    assert "roundtrip: OK" in text


def test_stores_endpoint_returns_409_with_failing_key(client, redis_client):
    good = LocalStore(name="good-store")
    good.set("ok", {"fine": True}, namespace="ns")

    bad = RedisStore(name="bad-store", store_id="broken", client=redis_client)
    bad.set("ok-key", b"value", namespace="ns")
    # Inject garbage bytes that the pickle codec cannot decode.
    redis_client.set("diffsync:broken:ns:broken-key", b"\x80\x04not-a-real-pickle")

    response = client.get("/stores")
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "conflict"
    bad_entry = next(item for item in body["stores"] if item["type"] == "RedisStore" and "broken" in item["name"])
    assert bad_entry["roundtrip_ok"] is False
    failed = bad_entry["roundtrip_failures"]
    assert [item["key"] for item in failed] == ["broken-key"]
    assert failed[0]["namespace"] == "ns"
    assert "error" in failed[0]

    text_response = client.get("/stores?format=text")
    assert text_response.status_code == 409
    assert "ns:broken-key" in text_response.text
    assert "roundtrip: FAILED" in text_response.text


def test_registry_tracks_live_stores_only():
    store = LocalStore(name="ephemeral")
    assert any(item.name == "ephemeral" for item in get_registered_stores())
    store.close()
