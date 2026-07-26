"""Tests for the HTTP layer's rate limiting and backoff.

These exist because of a real incident. An overnight run exhausted the Genius search
quota and then degraded to zero successful fetches for six hours: each worker retried
on its own schedule while the others kept issuing fresh requests, so the server never
stopped seeing traffic and the block never lifted. The fix was to make the penalty
global across threads and to give callers a way to detect a closed door.
"""

from __future__ import annotations

import threading
import time

import pytest

from music_tastes import http


@pytest.fixture(autouse=True)
def _reset_state():
    http._penalty.clear()
    http._last_request.clear()
    yield
    http._penalty.clear()
    http._last_request.clear()


class TestAdaptiveBackoff:
    def test_penalty_starts_empty(self):
        assert http._current_interval("api.genius.com") == pytest.approx(
            http.RATE_LIMITS["api.genius.com"]
        )

    def test_penalty_grows_multiplicatively(self):
        first = http.note_rate_limited("api.genius.com")
        second = http.note_rate_limited("api.genius.com")
        assert second > first

    def test_penalty_is_capped(self):
        for _ in range(50):
            http.note_rate_limited("api.genius.com")
        assert http._penalty["api.genius.com"] <= http.MAX_PENALTY

    def test_retry_after_header_respected(self):
        # A server-supplied Retry-After must win when it is longer than our own
        # backoff, otherwise we retry before the server is ready.
        delay = http.note_rate_limited("api.genius.com", retry_after=90)
        assert delay == pytest.approx(90)

    def test_success_decays_penalty(self):
        http.note_rate_limited("api.genius.com")
        before = http._penalty["api.genius.com"]
        http.note_success("api.genius.com")
        assert http._penalty["api.genius.com"] < before

    def test_success_on_clean_host_is_noop(self):
        http.note_success("example.com")
        assert "example.com" not in http._penalty


class TestCircuitBreaker:
    def test_closed_by_default(self):
        assert not http.circuit_open("api.genius.com")

    def test_opens_when_penalty_saturates(self):
        for _ in range(50):
            http.note_rate_limited("api.genius.com")
        assert http.circuit_open("api.genius.com")


class TestThrottle:
    def test_serial_requests_are_spaced(self):
        host = "musicbrainz.org"
        start = time.monotonic()
        http._throttle(host)
        http._throttle(host)
        elapsed = time.monotonic() - start
        assert elapsed >= http.RATE_LIMITS[host] * 0.9

    def test_concurrent_workers_share_one_budget(self):
        """The bug that caused the outage: per-thread throttling lets N threads
        issue N requests at once. Reservations must be global."""
        host = "musicbrainz.org"
        n = 4
        start = time.monotonic()
        threads = [threading.Thread(target=http._throttle, args=(host,)) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # n requests at 1 per interval must take at least (n-1) intervals.
        assert elapsed >= http.RATE_LIMITS[host] * (n - 1) * 0.8
