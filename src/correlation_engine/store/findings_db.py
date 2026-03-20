"""FindingsDatabase — Parquet-based persistence and query API for scan results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from correlation_engine.discovery.findings import Finding

logger = logging.getLogger(__name__)


class FindingsDatabase:
    """Persist and query scan findings stored as Parquet files.

    Each scan run produces one Parquet file.  A lightweight JSON index
    tracks all scan runs for fast lookup without reading every file.

    Parameters
    ----------
    base_path : str or Path
        Directory to store scan Parquet files and the index.
    """

    def __init__(self, base_path: str | Path = "data/findings"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_path / "index.json"

    # ── write ─────────────────────────────────────────────────────

    def save_findings(
        self,
        findings: list[Finding],
        scan_id: str,
        scanned_at: str,
    ) -> Path:
        """Persist findings as a Parquet file and update the index.

        Returns the path to the written Parquet file.
        """
        ts = scanned_at.replace(":", "-").replace("+", "_")[:19]
        filename = f"scan_{ts}.parquet"
        parquet_path = self.base_path / filename

        rows = [f.to_dict() for f in findings]
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)

        # Update index
        index = self._read_index()
        top_score = max((f.interestingness_score for f in findings), default=0.0)
        index["scans"].append({
            "scan_id": scan_id,
            "timestamp": scanned_at,
            "n_findings": len(findings),
            "top_score": round(top_score, 4),
            "parquet_file": filename,
        })
        self._write_index(index)

        logger.info(
            "Saved %d findings to %s (top score: %.4f).",
            len(findings), parquet_path, top_score,
        )
        return parquet_path

    # ── read ──────────────────────────────────────────────────────

    def load_latest(self, n: int = 50) -> list[Finding]:
        """Return the top-N findings from the most recent scan, sorted by score."""
        index = self._read_index()
        if not index["scans"]:
            return []

        latest = index["scans"][-1]
        path = self.base_path / latest["parquet_file"]
        if not path.exists():
            return []

        df = pd.read_parquet(path)
        df = df.sort_values("interestingness_score", ascending=False).head(n)
        return [Finding.from_dict(row) for _, row in df.iterrows()]

    def load_scan(self, scan_id: str) -> list[Finding]:
        """Load all findings for a specific scan_id."""
        index = self._read_index()
        for scan in index["scans"]:
            if scan["scan_id"] == scan_id:
                path = self.base_path / scan["parquet_file"]
                if not path.exists():
                    return []
                df = pd.read_parquet(path)
                return [Finding.from_dict(row) for _, row in df.iterrows()]
        return []

    def load_pair_history(
        self,
        series_a: str,
        series_b: str,
    ) -> pd.DataFrame:
        """Load all historical findings for a specific pair across all scans.

        Handles pair ordering: matches (a, b) in either order.
        """
        canon_a, canon_b = sorted([series_a, series_b])
        index = self._read_index()
        if not index["scans"]:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for scan in index["scans"]:
            path = self.base_path / scan["parquet_file"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            # Canonicalize stored pairs for comparison
            mask = df.apply(
                lambda row: sorted([row["series_a"], row["series_b"]]) == [canon_a, canon_b],
                axis=1,
            )
            matched = df[mask]
            if not matched.empty:
                frames.append(matched)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values("scanned_at")

    def was_seen_before(
        self,
        series_a: str,
        series_b: str,
        lookback_scans: int = 3,
    ) -> bool:
        """Check if a pair appeared in any of the last N scans."""
        canon_a, canon_b = sorted([series_a, series_b])
        index = self._read_index()
        recent = index["scans"][-lookback_scans:]

        for scan in recent:
            path = self.base_path / scan["parquet_file"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                if sorted([row["series_a"], row["series_b"]]) == [canon_a, canon_b]:
                    return True
        return False

    def load_all_scans(self) -> pd.DataFrame:
        """Return a DataFrame of scan metadata from the index."""
        index = self._read_index()
        if not index["scans"]:
            return pd.DataFrame(
                columns=["scan_id", "timestamp", "n_findings", "top_score", "parquet_file"],
            )
        return pd.DataFrame(index["scans"])

    def get_scan_count(self) -> int:
        """Number of completed scans."""
        return len(self._read_index()["scans"])

    # ── internals ─────────────────────────────────────────────────

    def _read_index(self) -> dict:
        if not self._index_path.exists():
            return {"scans": []}
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_index(self, index: dict) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
