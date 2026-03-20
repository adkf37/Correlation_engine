"""Parquet-based data cache with TTL expiry."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

_DEFAULT_CACHE_DIR = Path("data/cache")
_DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


class DataCache:
    """Cache DataFrames as Parquet files with time-to-live expiry.

    Parameters
    ----------
    cache_dir : Path or str
        Directory to store cached Parquet files.
    max_age : int
        Maximum age in seconds before a cached entry is considered stale.
    """

    def __init__(
        self,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        max_age: int = _DEFAULT_MAX_AGE_SECONDS,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, loader_type: str, **params) -> str:
        """Generate a deterministic cache key from loader type and parameters."""
        raw = json.dumps({"loader": loader_type, **params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> pd.DataFrame | None:
        """Retrieve a cached DataFrame, or None if missing/expired."""
        path = self._path_for(key)
        if not path.exists():
            return None

        age = time.time() - path.stat().st_mtime
        if age > self.max_age:
            return None

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Store a DataFrame in the cache."""
        path = self._path_for(key)
        df.to_parquet(path)

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if it existed."""
        path = self._path_for(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Remove all cached entries. Returns the number removed."""
        count = 0
        for p in self.cache_dir.glob("*.parquet"):
            p.unlink()
            count += 1
        return count

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"
