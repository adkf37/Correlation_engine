"""Finding dataclass — the core data record for every discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Finding:
    """A single discovered correlation relationship between two series."""

    scan_id: str
    scanned_at: str  # ISO 8601

    series_a: str
    series_b: str
    series_a_name: str
    series_b_name: str

    correlation: float
    correlation_method: str = "pearson"

    optimal_lag: int = 0
    lag_correlation: float = 0.0

    granger_p_value: float | None = None
    granger_direction: str | None = None  # "a_causes_b", "b_causes_a", "bidirectional"

    rolling_zscore: float = 0.0
    regime_change_detected: bool = False

    trigger_types: list[str] = field(default_factory=list)
    interestingness_score: float = 0.0
    is_new: bool = False

    template_summary: str = ""
    llm_summary: str | None = None

    lookback_days: int = 0
    frequency: str = "M"

    # ── serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to a flat dict suitable for DataFrame / Parquet storage.

        ``trigger_types`` is stored as a pipe-separated string.
        """
        d = asdict(self)
        d["trigger_types"] = "|".join(d["trigger_types"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Finding:
        """Reconstruct a Finding from a flat dict (e.g. Parquet row)."""
        d = dict(d)  # don't mutate the original
        raw = d.get("trigger_types", "")
        if isinstance(raw, str):
            d["trigger_types"] = [t for t in raw.split("|") if t]
        # handle numpy / pandas scalar types
        for int_field in ("optimal_lag", "lookback_days"):
            if int_field in d:
                d[int_field] = int(d[int_field])
        for float_field in (
            "correlation", "lag_correlation", "rolling_zscore",
            "interestingness_score",
        ):
            if float_field in d and d[float_field] is not None:
                d[float_field] = float(d[float_field])
        if "granger_p_value" in d:
            v = d["granger_p_value"]
            import math
            if v is None or (isinstance(v, float) and math.isnan(v)):
                d["granger_p_value"] = None
            else:
                d["granger_p_value"] = float(v)
        if "regime_change_detected" in d:
            d["regime_change_detected"] = bool(d["regime_change_detected"])
        if "is_new" in d:
            d["is_new"] = bool(d["is_new"])
        return cls(**d)
