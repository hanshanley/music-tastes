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

# Adaptive per-host penalty applied on top of RATE_LIMITS.
#
# Without this, a 429 is handled by each worker independently: every thread retries on
# its own backoff schedule while the others keep issuing fresh requests, so the server
# keeps seeing traffic and never lets us out of the penalty box. An overnight run
# degraded from 2.6 requests/second to zero successful fetches this way. The penalty
# below is global, so one 429 slows every thread, and it decays only on success.
_penalty: dict[str, float] = {}
MAX_PENALTY = 60.0


def circuit_open(host: str) -> bool:
    """True when a host has been rate limited to the point of being unusable.

    Once the adaptive penalty saturates, further requests are near-certain to return
    429 and merely extend the block. Callers with an alternative route should check
    this and take it instead of continuing to poll a closed door.
    """
    return _penalty.get(host, 0.0) >= MAX_PENALTY


def _current_interval(host: str) -> float:
    base = RATE_LIMITS.get(host, DEFAULT_RATE_LIMIT)
    return base + _penalty.get(host, 0.0)


def note_rate_limited(host: str, retry_after: float | None = None) -> float:
    """Register a 429 for a host and return how long to sleep before retrying."""
    with _throttle_lock:
        current = _penalty.get(host, 0.0)
        # Multiplicative increase, starting from the base interval.
        bumped = max(current * 2.0, RATE_LIMITS.get(host, DEFAULT_RATE_LIMIT))
        _penalty[host] = min(bumped, MAX_PENALTY)
        penalty = _penalty[host]
    return max(retry_after or 0.0, penalty)


def note_success(host: str) -> None:
    """Decay a host's penalty after a clean response."""
    if not _penalty.get(host):
        return
    with _throttle_lock:
        if _penalty.get(host):
            _penalty[host] *= 0.8
            if _penalty[host] < 0.01:
                _penalty.pop(host, None)


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
    how many threads are running, including any adaptive 429 penalty.
    """
    with _throttle_lock:
        interval = RATE_LIMITS.get(host, DEFAULT_RATE_LIMIT) + _penalty.get(host, 0.0)
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

        if r.status_code == 429:
            retry_after = None
            try:
                retry_after = float(r.headers.get("Retry-After", ""))
            except ValueError:
                retry_after = None
            delay = note_rate_limited(host, retry_after)
            last_error = RuntimeError(
                f"HTTP 429 from {host}; backing off {delay:.1f}s "
                f"(host penalty now {_penalty.get(host, 0):.1f}s)"
            )
            time.sleep(delay)
            continue

        if r.status_code >= 500:
            last_error = RuntimeError(f"HTTP {r.status_code} from {host}")
            time.sleep(2**attempt)
            continue

        note_success(host)
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
