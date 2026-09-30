"""Tests for the pluggable codecs."""

import pickle

from diffsync import DiffSyncModel
from diffsync.store import JSONCodec, PickleCodec


class _CodecModel(DiffSyncModel):
    _modelname = "codec_model"
    _identifiers = ("name",)

    name: str


def test_pickle_codec_roundtrip():
    codec = PickleCodec()
    assert codec.name == "pickle"
    obj = _CodecModel(name="a")
    obj.adapter = object()  # adapter must never be persisted
    payload = codec.encode(obj)
    decoded = codec.decode(payload)
    assert decoded.adapter is None
    assert decoded.dict() == obj.dict()


def test_pickle_codec_legacy_bytes_compatible():
    """Payloads produced the old way (manual copy + pickle.dumps) read back."""
    import copy

    obj = _CodecModel(name="legacy")
    clone = copy.copy(obj)
    clone.adapter = None
    legacy_bytes = pickle.dumps(clone)

    assert PickleCodec().decode(legacy_bytes) == obj


def test_json_codec_roundtrip():
    codec = JSONCodec()
    assert codec.name == "json"
    value = {"b": 2, "a": [1, 2, 3]}
    payload = codec.encode(value)
    assert isinstance(payload, bytes)
    assert codec.decode(payload) == value
