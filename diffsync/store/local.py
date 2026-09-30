"""LocalStore module: a ``shelve`` backed implementation of the template Store."""

from __future__ import annotations

import os
import shelve
import tempfile
from dbm import dumb as _dumb_db
from typing import Any, Iterator, Optional

from diffsync.store import Codec, PickleCodec, Store


class _LocalPickleCodec(PickleCodec):
    """Pickle codec for the local backend.

    Payloads are pickle bytes compatible with the historical local store;
    LocalStore keeps a session-level identity cache for live objects on top of
    the shelve database.
    """

    name = "pickle"


class LocalStore(Store):
    """File-backed local store using :mod:`shelve`.

    Args:
        path: Location of the shelve database (without extension). When None a
            temporary database is lazily created for this instance and removed
            on :meth:`close`.
        codec: Pluggable :class:`~diffsync.store.Codec`; defaults to the
            pickle codec that is byte-compatible with the previous local store.

    All other constructor arguments (``adapter``, ``name``, ...) keep their
    historical signatures and defaults.
    """

    default_codec: Codec = _LocalPickleCodec()

    def __init__(self, *args: Any, path: Optional[str] = None, **kwargs: Any) -> None:
        """Init method for LocalStore."""
        self._path = path
        self._tempdir: Optional[str] = None
        self._objects: dict = {}
        if path is not None:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
        super().__init__(*args, **kwargs)

    def _open(self) -> Any:
        """Open (lazily creating) the shelve database."""
        if self._path is None:
            self._tempdir = tempfile.mkdtemp(prefix="diffsync-localstore-")
            self._path = os.path.join(self._tempdir, "store")
        # writeback=False: writes persist immediately and no in-memory copy of
        # stored objects is retained, which matches a "落盘" backend.
        # dbm.dumb is a pure-Python backend without SQLite thread-affinity,
        # so the same database can be inspected from another thread
        # (e.g. a FastAPI TestClient).
        return shelve.Shelf(_dumb_db.open(self._path, flag="c"), writeback=False)  # type: ignore[arg-type]

    def _close(self, connection: Any) -> None:
        """Sync and close the shelve database, cleaning up temp databases."""
        connection.close()
        if self._tempdir is not None:
            tempdir = self._tempdir
            self._tempdir = None
            for entry in os.listdir(tempdir):
                os.remove(os.path.join(tempdir, entry))
            os.rmdir(tempdir)

    def _raw_get(self, key: str) -> Optional[bytes]:
        """Return the on-disk payload under ``key`` or None.

        Live in-session objects are served directly by :meth:`_get_value`
        rather than re-encoded here.
        """
        return self.connection.get(key)

    def _inspect_payload(self, physical_key: str) -> Optional[bytes]:
        """Inspect live in-session objects when present, else the disk copy."""
        if physical_key in self._objects:
            return self.codec.encode(self._objects[physical_key])
        return self.connection.get(physical_key)

    def _raw_set(self, key: str, payload: bytes) -> None:
        """Persist the raw payload under ``key`` and sync to disk."""
        self.connection[key] = payload
        self.connection.sync()

    def _raw_exists(self, key: str) -> bool:
        """Return whether ``key`` is present in the database."""
        return key in self._objects or key in self.connection

    def _raw_delete(self, key: str) -> bool:
        """Delete ``key`` from the database and sync; report prior presence."""
        existed = key in self._objects or key in self.connection
        self._objects.pop(key, None)
        if key in self.connection:
            del self.connection[key]
            self.connection.sync()
        return existed

    def _raw_scan(self, pattern: str) -> Iterator[str]:
        """Yield stored keys matching the glob ``pattern``.

        Live (in-session) keys come first in insertion order; additional keys
        only present on disk follow.
        """
        import fnmatch

        seen = set()
        for key in self._objects.keys():
            if fnmatch.fnmatchcase(key, pattern):
                seen.add(key)
                yield key
        for key in self.connection.keys():
            if key not in seen and fnmatch.fnmatchcase(key, pattern):
                seen.add(key)
                yield key

    def _get_value(
        self,
        namespace: str,
        key: str,
        *,
        allow_empty_key: bool = False,
        allow_empty_namespace: bool = False,
    ) -> Any:
        """Return the live instance within a session, else decode from shelve."""
        physical_key = self._physical_key(
            namespace,
            key,
            allow_empty_key=allow_empty_key,
            allow_empty_namespace=allow_empty_namespace,
        )
        if physical_key in self._objects:
            return self._objects[physical_key]
        return super()._get_value(
            namespace,
            key,
            allow_empty_key=allow_empty_key,
            allow_empty_namespace=allow_empty_namespace,
        )

    def set(self, key: str, value: Any, *, namespace: str, allow_empty_key: bool = False) -> None:
        """Retain ``value`` as the session's live instance and persist it.

        The in-session cache is the authoritative store within a process; the
        shelve write is a best-effort durability layer so non-picklable objects
        do not prevent normal adapter operation.
        """
        physical_key = self._physical_key(namespace, key, allow_empty_key=allow_empty_key)
        self._objects[physical_key] = value
        try:
            payload = self.codec.encode(value)
            self._raw_set(physical_key, payload)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "Unable to persist object to shelve; keeping the in-session copy", key=physical_key, error=str(exc)
            )
