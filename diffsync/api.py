"""Read-only HTTP API exposing the configured storage backends.

The single ``GET /stores`` endpoint lists every live :class:`~diffsync.store.Store`
with its codec, key count and a codec round-trip verification. It is meant to be
human-readable for manual/screen-recording checks; a plain-text rendering is
available through ``GET /stores?format=text``.

If any stored payload fails the round-trip check, the endpoint answers with
HTTP 409 and explicitly lists the failing namespace/key pairs (never a bare
"OK").
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from diffsync.store import get_registered_stores

app = FastAPI(title="DiffSync stores API", version="1.0")


def inspect_stores() -> List[Dict[str, Any]]:
    """Collect the codec/round-trip inspection report of all live stores."""
    return [store.inspect() for store in get_registered_stores()]


def render_text(reports: List[Dict[str, Any]]) -> str:
    """Render the inspection report as readable plain text."""
    lines = ["DiffSync stores", "===============", f"stores: {len(reports)}", ""]
    for report in reports:
        status = "OK" if report["roundtrip_ok"] else "FAILED"
        lines.extend(
            [
                f"- name: {report['name']}",
                f"  type: {report['type']}",
                f"  codec: {report['codec']}",
                f"  keys: {report['key_count']}",
                f"  roundtrip: {status}",
            ]
        )
        if not report["roundtrip_ok"]:
            lines.append("  failed keys:")
            for failure in report["roundtrip_failures"]:
                lines.append(f"    * {failure['namespace']}:{failure['key']} -> {failure['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@app.get("/stores")
def stores(format: str = Query(default="json", pattern="^(json|text)$")) -> Response:
    """List stores with codec, round-trip result and key count.

    Returns 409 with the offending keys explicitly listed when any round-trip
    verification fails.
    """
    reports = inspect_stores()
    all_ok = all(report["roundtrip_ok"] for report in reports)
    status_code = 200 if all_ok else 409
    if format == "text":
        return PlainTextResponse(render_text(reports), status_code=status_code)
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "conflict",
            "stores": reports,
            "total_stores": len(reports),
            "total_keys": sum(report["key_count"] for report in reports),
        },
    )
