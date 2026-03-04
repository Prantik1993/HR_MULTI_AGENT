import hashlib
from typing import Optional


class QueryCache:
    def __init__(self, max_size: int = 500):
        self._store: dict[str, str] = {}
        self._max_size = max_size

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        return self._store.get(self._key(query))

    def set(self, query: str, response: str) -> None:
        if len(self._store) >= self._max_size:
            del self._store[next(iter(self._store))]
        self._store[self._key(query)] = response

    def __len__(self) -> int:
        return len(self._store)


cache = QueryCache()
