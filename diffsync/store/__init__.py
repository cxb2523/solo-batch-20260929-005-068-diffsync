"""Base store module with a template-method ``Store`` and pluggable ``Codec``.

The :class:`Store` base class centralises the physical key layout
(``root:namespace:key``), the encoding/decoding of values and the
connection/context-manager lifecycle. Backends only implement the raw
entity-read/write hooks (``_raw_get``, ``_raw_set``, ``_raw_exists``,
``_raw_delete`` and ``_raw_scan``) plus, optionally, ``_open``/``_close``.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union, cast
from weakref import WeakSet

import structlog  # type: ignore

from diffsync.exceptions import ObjectAlreadyExists, ObjectNotFound

if TYPE_CHECKING:
    from diffsync import Adapter, DiffSyncModel

# Separator used in the physical keys of every backend. It is also the
# forbidden character for namespaces and keys, so that no crafted value can
# ever be confused for a separator (no migration or version prefix required).
KEY_SEPARATOR = ":"
# Characters that would turn a namespace/key into a glob when scanning.
_FORBIDDEN_NAME_CHARS = (":", "*", "?", "[", "]")


class Codec(ABC):
    """Abstract encode/decode pair used by a :class:`Store`.

    Implementations convert Python objects into ``bytes`` payloads (and back),
    so that backend implementations never need to know how values are
    serialised.
    """

    #: Short, human-readable identifier reported by the ``/stores`` endpoint.
    name: str = "codec"

    @abstractmethod
    def encode(self, value: Any) -> bytes:
        """Serialise ``value`` into bytes ready for physical storage."""

    @abstractmethod
    def decode(self, payload: bytes) -> Any:
        """Deserialise a physical ``bytes`` payload back into a Python object."""


class PickleCodec(Codec):
    """Default codec: binary pickle payloads.

    This is the exact wire format historically used by both the local and the
    Redis backends, so data previously written by either backend reads back
    without any migration or version marker.
    """

    name = "pickle"

    def encode(self, value: Any) -> bytes:
        """Pickle ``value``; the ``adapter`` back-reference is never persisted."""
        obj_copy = copy.copy(value)
        if hasattr(obj_copy, "adapter"):
            obj_copy.adapter = None
        return pickle_dumps(obj_copy)

    def decode(self, payload: bytes) -> Any:
        """Unpickle ``payload``."""
        return pickle_loads(payload)  # noqa: S301


class JSONCodec(Codec):
    """Optional JSON codec for plain JSON-serialisable values."""

    name = "json"

    def encode(self, value: Any) -> bytes:
        """Serialise ``value`` to UTF-8 JSON."""
        return json.dumps(value, sort_keys=True).encode("utf-8")

    def decode(self, payload: bytes) -> Any:
        """Deserialise a UTF-8 JSON payload."""
        return json.loads(payload.decode("utf-8"))


_REGISTERED_STORES: WeakSet = WeakSet()


def get_registered_stores() -> List["Store"]:
    """Return the currently live stores known to the store registry."""
    return list(_REGISTERED_STORES)


def _clear_registered_stores() -> None:
    """Reset the store registry (mainly used by tests)."""
    _REGISTERED_STORES.clear()


__all__ = [
    "Store",
    "BaseStore",
    "Codec",
    "PickleCodec",
    "JSONCodec",
    "KEY_SEPARATOR",
    "get_registered_stores",
]


class Store:
    """Template-method store shared by all storage backends.

    The public key/value API (:meth:`get`, :meth:`set`, :meth:`exists`,
    :meth:`delete`, :meth:`keys`) handles namespace/key validation, physical
    key prefix assembly and codec-based (de)serialisation; concrete backends
    only provide the five ``_raw_*`` hooks and, when needed, ``_open`` /
    ``_close`` for the connection lifecycle.
    """

    #: Default codec injected by every backend unless overridden.
    default_codec: Codec = PickleCodec()

    def __init__(
        self,
        *args: Any,  # pylint: disable=unused-argument
        adapter: Optional["Adapter"] = None,
        name: str = "",
        codec: Optional[Codec] = None,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Init method for Store."""
        self.adapter = adapter
        self.name = name or self.__class__.__name__
        self.codec: Codec = codec if codec is not None else self.default_codec
        self._connection: Any = None
        self._closed = False
        self._log = structlog.get_logger().new(store=self)
        self.connect()
        _REGISTERED_STORES.add(self)

    def __str__(self) -> str:
        """Render store name."""
        return self.name

    def __enter__(self) -> "Store":
        """Open/ensure the connection when used as a context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the connection on context-manager exit."""
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude live connections and logger from pickled state."""
        state = dict(self.__dict__)
        state["_connection"] = None
        state.pop("_log", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickled state, rebuilding the logger lazily."""
        self.__dict__.update(state)
        if not hasattr(self, "_log"):
            self._log = structlog.get_logger().new(store=self)

    # ------------------------------------------------------------------
    # Connection lifecycle, taken over by the base class
    # ------------------------------------------------------------------
    def connect(self) -> "Store":
        """Idempotently open the backend connection.

        Subclasses implement :meth:`_open` to build the actual connection.
        """
        if self._connection is None:
            self._connection = self._open()
            self._closed = False
        return self

    def close(self) -> None:
        """Idempotently close the backend connection via :meth:`_close`."""
        if self._connection is not None:
            try:
                self._close(self._connection)
            finally:
                self._connection = None
                self._closed = True

    @property
    def connection(self) -> Any:
        """Active backend connection, reconnecting lazily after a ``close``."""
        if self._connection is None:
            self.connect()
        return self._connection

    def _open(self) -> Any:
        """Hook: build and return the backend connection.

        Backends without an explicit connection (in-memory dictionaries) can
        leave the default implementation in place.
        """
        return {}

    def _close(self, connection: Any) -> None:  # pylint: disable=unused-argument
        """Hook: release a previously opened backend connection."""

    # ------------------------------------------------------------------
    # Key layout, shared by every backend
    # ------------------------------------------------------------------
    @property
    def _prefix_parts(self) -> Tuple[str, ...]:
        """Backend-specific root prefix segments, preceding the namespace."""
        return ()

    def _validate_name(self, value: Any, *, what: str, allow_empty: bool = False) -> str:
        """Validate a namespace or key *before* anything touches the backend."""
        if not isinstance(value, str) or (not value and not allow_empty):
            raise ValueError(f"Invalid {what} {value!r}: must be a non-empty string")
        for char in _FORBIDDEN_NAME_CHARS:
            if char in value:
                raise ValueError(f"Invalid {what} {value!r}: must not contain {char!r}")
        return value

    def _physical_key(
        self,
        namespace: str,
        key: str,
        *,
        allow_empty_key: bool = False,
        allow_empty_namespace: bool = False,
    ) -> str:
        """Assemble the full physical key from prefix, namespace and key."""
        namespace = self._validate_name(namespace, what="namespace", allow_empty=allow_empty_namespace)
        key = self._validate_name(key, what="key", allow_empty=allow_empty_key)
        return KEY_SEPARATOR.join((*self._prefix_parts, namespace, key))

    def _physical_pattern(self, namespace: Optional[str]) -> str:
        """Assemble the scan glob for a namespace (or the whole store)."""
        parts = (*self._prefix_parts, "*")
        if namespace is not None:
            namespace = self._validate_name(namespace, what="namespace")
            parts = (*self._prefix_parts, namespace, "*")
        return KEY_SEPARATOR.join(parts)

    @staticmethod
    def _decode_key(raw_key: Any) -> str:
        """Normalise a physical key (``bytes`` or ``str``) to ``str``."""
        if isinstance(raw_key, bytes):
            return raw_key.decode("utf-8")
        return raw_key

    def _split_physical_key(self, raw_key: Any) -> Tuple[Optional[str], Optional[str]]:
        """Split a physical key into ``(namespace, key)`` within this store."""
        key = self._decode_key(raw_key)
        skip = len(self._prefix_parts)
        segments = key.split(KEY_SEPARATOR)
        remainder = segments[skip:]
        if len(remainder) >= 2:
            return remainder[0], remainder[-1]
        if len(remainder) == 1:
            return None, remainder[0]
        return None, None

    # ------------------------------------------------------------------
    # Raw backend hooks, the only pieces a concrete backend must provide
    # ------------------------------------------------------------------
    def _raw_get(self, key: str) -> Optional[bytes]:
        """Return the raw ``bytes`` payload for ``key`` or ``None``."""
        raise NotImplementedError

    def _raw_set(self, key: str, payload: bytes) -> None:
        """Persist the raw ``bytes`` payload under ``key``."""
        raise NotImplementedError

    def _raw_exists(self, key: str) -> bool:
        """Return whether a payload is currently stored under ``key``."""
        raise NotImplementedError

    def _raw_delete(self, key: str) -> bool:
        """Delete ``key``; return whether it existed beforehand."""
        raise NotImplementedError

    def _raw_scan(self, pattern: str) -> Iterator[Any]:
        """Yield physical keys matching the glob ``pattern``."""
        raise NotImplementedError

    def _inspect_payload(self, physical_key: str) -> Optional[bytes]:
        """Return the payload to verify during :meth:`inspect`.

        Defaults to the physical payload; backends with an authoritative
        in-session view may override this to inspect live objects.
        """
        return self._raw_get(physical_key)

    # ------------------------------------------------------------------
    # Generic key/value template API
    # ------------------------------------------------------------------
    def get(
        self, key: Optional[str] = None, *, namespace: Optional[str] = None, model: Any = None, identifier: Any = None
    ) -> Any:  # noqa: E501
        """Get a value by key (generic KV API) or a model entity (legacy API).

        ``get(key, namespace=...)`` returns the decoded value stored under
        ``namespace:key`` or ``None`` if absent.
        ``get(model=..., identifier=...)`` returns a :class:`DiffSyncModel` and
        raises :class:`ObjectNotFound` when missing, as before.
        """
        if model is not None or identifier is not None:
            return self._get_entity(model=model, identifier=identifier)
        if key is None:
            raise ValueError("get() requires either 'key' or 'model'/'identifier' arguments")
        if namespace is None:
            raise ValueError("get(key=...) requires an explicit 'namespace' argument")
        return self._get_value(namespace, key)

    def _get_value(
        self,
        namespace: str,
        key: str,
        *,
        allow_empty_key: bool = False,
        allow_empty_namespace: bool = False,
    ) -> Any:
        """Decode the payload stored under ``namespace:key`` if present."""
        payload = self._raw_get(
            self._physical_key(
                namespace,
                key,
                allow_empty_key=allow_empty_key,
                allow_empty_namespace=allow_empty_namespace,
            )
        )
        if payload is None:
            return None
        return self.codec.decode(payload)

    def set(self, key: str, value: Any, *, namespace: str, allow_empty_key: bool = False) -> None:
        """Encode ``value`` and store it under ``namespace:key``.

        Namespace and key are validated before any backend write.
        """
        physical_key = self._physical_key(namespace, key, allow_empty_key=allow_empty_key)
        self._raw_set(physical_key, self.codec.encode(value))

    def exists(self, key: str, *, namespace: str, allow_empty_key: bool = False) -> bool:
        """Return whether ``namespace:key`` is present in the store."""
        return self._raw_exists(self._physical_key(namespace, key, allow_empty_key=allow_empty_key))

    def delete(
        self,
        key: str,
        *,
        namespace: str,
        missing_ok: bool = True,
        allow_empty_key: bool = False,
    ) -> None:
        """Delete ``namespace:key``.

        Raises :class:`ObjectNotFound` when the key is absent and
        ``missing_ok`` is False.
        """
        physical_key = self._physical_key(namespace, key, allow_empty_key=allow_empty_key)
        if not self._raw_delete(physical_key) and not missing_ok:
            raise ObjectNotFound(f"{namespace} {key} not present in {str(self)}")

    def keys(self, *, namespace: Optional[str] = None) -> Union[List[Tuple[str, str]], List[str]]:
        """List keys present in the store.

        With ``namespace`` set, returns the plain keys inside that namespace;
        without it, returns ``(namespace, key)`` tuples for the whole store.
        """
        pattern = self._physical_pattern(namespace)
        if namespace is None:
            results: List[Tuple[str, str]] = []
            for raw in self._raw_scan(pattern):
                pair = self._split_physical_key(raw)
                if pair[0] is not None and pair[1] is not None:
                    results.append((pair[0], pair[1]))
            return list(dict.fromkeys(results))
        prefix_len = len(KEY_SEPARATOR.join((*self._prefix_parts, namespace, "")))
        plain_keys = [self._decode_key(raw)[prefix_len:] for raw in self._raw_scan(pattern)]
        return list(dict.fromkeys(plain_keys))

    # ------------------------------------------------------------------
    # DiffSyncModel entity API, expressed through the template primitives
    # ------------------------------------------------------------------
    def get_all_model_names(self) -> Set[str]:
        """Get all the model names (namespaces) stored."""
        all_keys = cast(List[Tuple[str, str]], self.keys(namespace=None))
        return {namespace for namespace, _key in all_keys}

    def _get_entity(
        self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]], identifier: Union[str, Dict]
    ) -> "DiffSyncModel":
        """Get one model object from the data store based on its unique id."""
        object_class, modelname = self._get_object_class_and_model(model)
        uid = self._get_uid(model, object_class, identifier)
        obj = self._get_value(modelname, uid, allow_empty_key=True, allow_empty_namespace=True)
        if obj is None:
            raise ObjectNotFound(f"{modelname} {uid} not present in {str(self)}")
        obj.adapter = self.adapter
        return obj

    def get_all(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]) -> List["DiffSyncModel"]:
        """Get all objects of a given type."""
        modelname = model if isinstance(model, str) else model.get_type()
        results = []
        plain_keys = cast(List[str], self.keys(namespace=modelname))
        for uid in plain_keys:
            obj = self._get_value(modelname, uid, allow_empty_key=True, allow_empty_namespace=True)
            if obj is not None:
                obj.adapter = self.adapter
                results.append(obj)
        return results

    def get_by_uids(
        self, *, uids: List[str], model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]
    ) -> List["DiffSyncModel"]:
        """Get multiple objects from the store by their unique IDs and type."""
        return [self._get_entity(model=model, identifier=uid) for uid in uids]

    def add(self, *, obj: "DiffSyncModel") -> None:
        """Add a DiffSyncModel object to the store.

        Raises:
            ObjectAlreadyExists: if a different object with the same uid is
                already present. Namespace/key validity is checked before the
                write so conflicts surface before anything reaches the backend.
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()
        # Validate the physical key before touching the backend.
        self._physical_key(modelname, uid, allow_empty_key=True)
        if self.exists(uid, namespace=modelname, allow_empty_key=True):
            existing_obj = self._get_value(modelname, uid, allow_empty_key=True, allow_empty_namespace=True)
            if existing_obj is obj:
                # Same instance: nothing to do.
                return
            if existing_obj.dict() == obj.dict():
                # Equal content (the historical RedisStore contract): no-op.
                return
            raise ObjectAlreadyExists(f"Object {uid} already present", obj)
        if not obj.adapter:
            obj.adapter = self.adapter
        self.set(uid, obj, namespace=modelname, allow_empty_key=True)

    def update(self, *, obj: "DiffSyncModel") -> None:
        """Update a DiffSyncModel object in the store (unconditional upsert)."""
        if not obj.adapter:
            obj.adapter = self.adapter
        self.set(obj.get_unique_id(), obj, namespace=obj.get_type(), allow_empty_key=True)

    def remove_item(self, modelname: str, uid: str) -> None:
        """Remove one item from store."""
        self.delete(uid, namespace=modelname, missing_ok=False, allow_empty_key=True)

    def remove(self, *, obj: "DiffSyncModel", remove_children: bool = False) -> None:
        """Remove a DiffSyncModel object from the store.

        Args:
            obj: object to remove
            remove_children: If True, also recursively remove any children of this object

        Raises:
            ObjectNotFound: if the object is not present
        """
        modelname = obj.get_type()
        uid = obj.get_unique_id()

        self.remove_item(modelname, uid)

        if obj.adapter:
            obj.adapter = None

        if remove_children:
            for child_type, child_fieldname in obj.get_children_mapping().items():
                for child_id in getattr(obj, child_fieldname):
                    try:
                        child_obj = self.get(model=child_type, identifier=child_id)
                        self.remove(obj=child_obj, remove_children=remove_children)
                    except ObjectNotFound:
                        # Cleanup code: log and continue instead of raising.
                        self._log.error(
                            "Unable to remove child element as it was not found!",
                            child_type=child_type,
                            child_id=child_id,
                            parent_type=modelname,
                            parent_id=uid,
                        )

    def count(self, *, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"], None] = None) -> int:
        """Return the number of elements of a model, or all elements if unspecified."""
        if model is None:
            return len(self.keys())
        modelname = model if isinstance(model, str) else model.get_type()
        return len(self.keys(namespace=modelname))

    # ------------------------------------------------------------------
    # Introspection used by the /stores API endpoint
    # ------------------------------------------------------------------
    def inspect(self) -> Dict[str, Any]:
        """Describe the codec, key count and round-trip check of this store.

        Every stored payload is decoded and re-encoded; values that differ from
        the original, or that fail to round-trip, are reported in
        ``roundtrip_failures``. The check is read-only.
        """
        failures: List[Dict[str, str]] = []
        key_count = 0
        all_pairs = cast(List[Tuple[str, str]], self.keys(namespace=None))
        for namespace, key in all_pairs:
            key_count += 1
            physical_key = self._physical_key(namespace, key)
            try:
                payload = self._inspect_payload(physical_key)
                if payload is None:
                    failures.append({"namespace": namespace, "key": key, "error": "key vanished during inspection"})
                    continue
                value = self.codec.decode(payload)
                reencoded = self.codec.encode(value)
                if not self._roundtrip_equal(payload, reencoded, value):
                    failures.append(
                        {"namespace": namespace, "key": key, "error": "re-encoded payload differs from original"}
                    )
            except Exception as exc:  # noqa: BLE001 - report any codec failure verbatim
                failures.append({"namespace": namespace, "key": key, "error": f"{type(exc).__name__}: {exc}"})
        return {
            "name": str(self),
            "type": self.__class__.__name__,
            "codec": self.codec.name,
            "key_count": key_count,
            "roundtrip_ok": not failures,
            "roundtrip_failures": failures,
        }

    def _roundtrip_equal(self, original: bytes, reencoded: bytes, value: Any) -> bool:
        """Compare two payloads, tolerating non-deterministic codecs (pickle)."""
        if original == reencoded:
            return True
        # Pickle bytes can differ for equal objects (memo/object ids); compare
        # a second decode of the re-encoded payload against the decoded value.
        try:
            return self.codec.decode(reencoded) == value
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Helpers kept for backwards compatibility
    # ------------------------------------------------------------------
    def get_or_instantiate(
        self, *, model: Type["DiffSyncModel"], ids: Dict, attrs: Optional[Dict] = None
    ) -> Tuple["DiffSyncModel", bool]:
        """Attempt to get the object or instantiate it with provided ids/attrs."""
        created = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            if not attrs:
                attrs = {}
            obj = model(**ids, **attrs)
            self.add(obj=obj)
            created = True

        return obj, created

    def get_or_add_model_instance(self, obj: "DiffSyncModel") -> Tuple["DiffSyncModel", bool]:
        """Attempt to get the object with provided obj identifiers or instantiate obj."""
        model = obj.get_type()
        ids = obj.get_unique_id()

        try:
            return self.get(model=model, identifier=ids), False
        except ObjectNotFound:
            self.add(obj=obj)
            return obj, True

    def update_or_instantiate(
        self, *, model: Type["DiffSyncModel"], ids: Dict, attrs: Dict
    ) -> Tuple["DiffSyncModel", bool]:
        """Attempt to update an existing object or instantiate it."""
        created = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            obj = model(**ids, **attrs)
            self.add(obj=obj)
            created = True

        for attr, value in attrs.items():
            if getattr(obj, attr) != value:
                setattr(obj, attr, value)

        return obj, created

    def update_or_add_model_instance(self, obj: "DiffSyncModel") -> Tuple["DiffSyncModel", bool]:
        """Attempt to update an existing object or instantiate obj."""
        model = obj.get_type()
        ids = obj.get_unique_id()
        attrs = obj.get_attrs()

        added = False
        try:
            obj = self.get(model=model, identifier=ids)
        except ObjectNotFound:
            self.add(obj=obj)
            added = True

        for attr, value in attrs.items():
            setattr(obj, attr, value)

        return obj, added

    def _get_object_class_and_model(
        self, model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]]
    ) -> Tuple[Union["DiffSyncModel", Type["DiffSyncModel"], None], str]:
        """Get object class and model name for a model."""
        if isinstance(model, str):
            modelname = model
            if not hasattr(self.adapter, model):
                return None, modelname
            object_class = getattr(self.adapter, model)
        else:
            object_class = model
            modelname = model.get_type()

        return object_class, modelname

    @staticmethod
    def _get_uid(
        model: Union[str, "DiffSyncModel", Type["DiffSyncModel"]],
        object_class: Union["DiffSyncModel", Type["DiffSyncModel"], None],
        identifier: Union[str, Dict],
    ) -> str:
        """Get the related uid for a model and an identifier."""
        if isinstance(identifier, str):
            uid = identifier
        elif object_class:
            uid = object_class.create_unique_id(**identifier)
        else:
            raise ValueError(
                f"Invalid args: ({model}, {object_class}, {identifier}): "
                f"either {object_class} should be a class/instance or {identifier} should be a str"
            )
        return uid


# Backwards-compatible public name; existing imports keep working unchanged.
BaseStore = Store
