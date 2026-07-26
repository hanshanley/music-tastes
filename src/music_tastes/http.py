"""HTTP fetching with on-disk caching and polite per-host rate limiting.

Every network stage in this project goes through :func:`get`. Responses are cached
on disk keyed by URL so re-running a stage costs nothing, and each host gets its own
minimum request interval so we stay inside published rate limits.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .paths import CACHE, user_agent

# Minimum seconds between requests, per host, enforced globally across all worker
# threads. MusicBrainz and AcousticBrainz publish a 1 req/s limit for anonymous
# clients; the others are courtesy values chosen to stay well inside what a large
# site tolerates from a single research client.
RATE_LIMITS = {
    "musicbrainz.org": 1.05,
    "acousticbrainz.org": 1.05,
    "api.reccobeats.com": 0.25,
    "api.genius.com": 0.34,
    "genius.com": 0.34,
    "api.getsongbpm.com": 0.5,
}
DEFAULT_RATE_LIMIT = 0.25

_last_request: dict[str, float] = {}
_throttle_lock = threading.Lock()
_cache_locks: dict[str, threading.Lock] = {}
_cache_locks_guard = threading.Lock()


@dataclass
class Response:
    """A fetched response plus the provenance we need to cite it later."""

    url: str
    status: int
    text: str
    retrieved_at: str
    from_cache: bool

    def json(self) -> Any:
        return json.loads(self.text)


def _cache_path(url: str, namespace: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:24]
    d = CACHE / namespace / digest[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def _throttle(host: str) -> None:
    """Block until this host's minimum interval has elapsed.

    The reservation is made while holding the lock so that concurrent workers queue
    up behind each other instead of all sleeping until the same instant and then
    firing simultaneously. Total request rate to a host is therefore capped no matter
    how many threads are running.
    """
    interval = RATE_LIMITS.get(host, DEFAULT_RATE_LIMIT)
    with _throttle_lock:
        now = time.monotonic()
        earliest = _last_request.get(host, 0.0) + interval
        wait = max(0.0, earliest - now)
        _last_request[host] = max(now, earliest)
    if wait:
        time.sleep(wait)


def get(
    url: str,
    *,
    namespace: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
    use_cache: bool = True,
    cache_errors: bool = True,
) -> Response:
    """Fetch ``url``, returning a cached copy when one exists.

    ``cache_errors`` stores 4xx responses too, so that a song genuinely absent from a
    source does not get re-requested on every run. 5xx and network errors are retried
    with exponential backoff and never cached.
    """
    full = requests.Request("GET", url, params=params).prepare().url
    path = _cache_path(full, namespace)

    if use_cache and path.exists():
        payload = json.loads(path.read_text())
        return Response(
            url=full,
            status=payload["status"],
            text=payload["text"],
            retrieved_at=payload["retrieved_at"],
            from_cache=True,
        )

    host = urlparse(full).netloc
    merged = {"User-Agent": user_agent(), "Accept": "application/json"}
    if headers:
        merged.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries):
        _throttle(host)
        try:
            r = requests.get(full, headers=merged, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
            continue

        if r.status_code >= 500 or r.status_code == 429:
            last_error = RuntimeError(f"HTTP {r.status_code} from {host}")
            # 429 means we are going too fast; back off harder than for a 5xx.
            time.sleep((2**attempt) * (4 if r.status_code == 429 else 1))
            continue

        resp = Response(
            url=full,
            status=r.status_code,
            text=r.text,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            from_cache=False,
        )
        if use_cache and (r.ok or cache_errors):
            path.write_text(
                json.dumps(
                    {
                        "url": full,
                        "status": resp.status,
                        "text": resp.text,
                        "retrieved_at": resp.retrieved_at,
                    }
                )
            )
        return resp

    raise RuntimeError(f"Failed to fetch {full} after {retries} attempts") from last_error
