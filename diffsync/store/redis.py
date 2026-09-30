"""RedisStore module.

The historical Redis wire format is preserved byte-for-byte:

* key layout: ``diffsync:<store_id>:<modelname>:<uid>``
* values: plain pickle payloads of the model with ``adapter`` unset

No migration is performed and no protocol/version marker is added.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

try:
    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
except ImportError as ierr:
    print("Redis is not installed. Have you installed diffsync with redis extra? `pip install diffsync[redis]`")
    raise ierr

from diffsync.exceptions import ObjectNotFound, ObjectStoreException
from diffsync.store import PickleCodec, Store

if TYPE_CHECKING:
    from diffsync import DiffSyncModel

REDIS_DIFFSYNC_ROOT_LABEL = "diffsync"


class RedisStore(Store):
    """Redis-backed store.

    The constructor signature is unchanged compared to the previous
    implementation; the Redis client itself is created lazily while the
    availability check (``ping``) keeps raising :class:`ObjectStoreException`
    at construction time as before. An extra ``client`` keyword may be passed
    to inject an already-constructed client (used by the fakeredis tests).
    """

    default_codec = PickleCodec()

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        store_id: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6379,
        url: Optional[str] = None,
        db: int = 0,
        client: Optional[Redis] = None,
        **kwargs: Any,
    ) -> None:
        """Init method for RedisStore."""
        if url and host and port:
            raise ValueError("'url' and 'host' arguments can't be specified together.")

        self._host = host
        self._port = port
        self._url = url
        self._db = db
        self._injected_client = client is not None
        self._client = client
        self._client_factory: Optional[Any] = None

        self._store_id = store_id if store_id else str(uuid.uuid4())
        self._store_label = f"{REDIS_DIFFSYNC_ROOT_LABEL}:{self._store_id}"

        # super().__init__() opens (or verifies) the connection eagerly.
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        """Render store name."""
        return f"{self.name} ({self._store_id})"

    @property
    def _prefix_parts(self) -> Tuple[str, ...]:
        """Root prefix segments: the historical ``diffsync:<store_id>`` label."""
        return (REDIS_DIFFSYNC_ROOT_LABEL, self._store_id)

    def _open(self) -> Redis:
        """Build the lazy Redis client and verify availability via ping."""
        if self._client is not None:
            client = self._client
        elif self._client_factory is not None:
            client = self._client_factory()
        elif self._url:
            client = Redis.from_url(self._url, db=self._db)
        elif self._host:
            client = Redis(host=self._host, port=self._port, db=self._db)
        else:
            raise ObjectStoreException("Redis store is unavailable.")
        try:
            if not client.ping():
                raise RedisConnectionError("Redis ping returned False")
        except RedisError as exc:
            raise ObjectStoreException("Redis store is unavailable.") from exc
        self._client = client
        if self._client_factory is None and not self._injected_client:
            self._client_factory = lambda: client
        return client

    def _close(self, connection: Any) -> None:
        """Close the client unless it was injected by the caller."""
        if not self._injected_client:
            connection.close()
        self._client = None

    def _raw_get(self, key: str) -> Optional[bytes]:
        """Fetch the raw payload under ``key``."""
        return self.connection.get(key)

    def _raw_set(self, key: str, payload: bytes) -> None:
        """Store the raw payload under ``key``."""
        self.connection.set(key, payload)

    def _raw_exists(self, key: str) -> bool:
        """Return whether ``key`` exists."""
        return bool(self.connection.exists(key))

    def _raw_delete(self, key: str) -> bool:
        """Delete ``key``; return whether it existed beforehand."""
        return bool(self.connection.delete(key))

    def _raw_scan(self, pattern: str) -> Iterator[Any]:
        """Yield keys matching ``pattern`` using the historical SCAN approach."""
        return self.connection.scan_iter(pattern)

    # ------------------------------------------------------------------
    # Backwards-compatible helpers from the original RedisStore
    # ------------------------------------------------------------------
    def _get_key_for_object(self, modelname: str, uid: str) -> str:
        """Return the historical physical Redis key for a model and uid."""
        return f"{self._store_label}:{modelname}:{uid}"

    def _get_object_from_redis_key(self, key: str) -> "DiffSyncModel":
        """Read and decode an object directly from a physical Redis key."""
        pickled_object = self.connection.get(key)
        if pickled_object:
            obj_result = self.codec.decode(pickled_object)
            obj_result.adapter = self.adapter
            return obj_result
        raise ObjectNotFound(f"{key} not present in Cache")
