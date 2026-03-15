"""
app/cache/query_cache.py
-------------------------
FIX [10]: Added TTL (time-to-live). Stale answers are now expired
          after `ttl_seconds` (default 1 hour). Re-ingesting documents
          will produce fresh answers after the TTL window.
"""
from __future__ import annotations
import hashlib
import time
from typing import Optional
from app.core.logger import get_logger

logger = get_logger(__name__)


class QueryCache:
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        # store: key → (response, expires_at)
        self._store: dict[str, tuple[str, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        k = self._key(query)
        entry = self._store.get(k)
        if entry is None:
            return None
        response, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[k]
            logger.debug("cache | TTL expired for key %s", k[:8])
            return None
        logger.debug("cache | HIT for key %s", k[:8])
        return response

    def set(self, query: str, response: str) -> None:
        if len(self._store) >= self._max_size:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            logger.debug("cache | evicted oldest entry (max_size=%d)", self._max_size)
        k = self._key(query)
        self._store[k] = (response, time.monotonic() + self._ttl)
        logger.debug("cache | SET key %s ttl=%ds", k[:8], self._ttl)

    def __len__(self) -> int:
        return len(self._store)


cache = QueryCache()
