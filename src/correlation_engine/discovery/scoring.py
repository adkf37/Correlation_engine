"""FindingScorer — evaluates 6 interestingness criteria and produces a composite score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCORING_CONFIG = _PROJECT_ROOT / "config" / "scoring.yaml"


@dataclass
class ScoringConfig:
    """Thresholds and weights loaded from ``scoring.yaml``."""

    correlation_threshold: float = 0.7
    zscore_threshold: float = 2.0
    granger_p_threshold: float = 0.05
    lag_correlation_threshold: float = 0.6

    weights: dict[str, float] = field(default_factory=lambda: {
        "high_correlation": 0.25,
        "newly_emerging": 0.15,
        "regime_change": 0.20,
        "granger_causality": 0.20,
        "anomalous_lag": 0.10,
        "rolling_divergence": 0.10,
    })

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_SCORING_CONFIG) -> ScoringConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        thresholds = raw.get("thresholds", {})
        weights = raw.get("weights", {})
        return cls(
            correlation_threshold=thresholds.get("correlation", 0.7),
            zscore_threshold=thresholds.get("zscore", 2.0),
            granger_p_threshold=thresholds.get("granger_p", 0.05),
            lag_correlation_threshold=thresholds.get("lag_correlation", 0.6),
            weights=weights if weights else cls.weights,
        )

    def validate(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}. "
                f"Weights: {self.weights}"
            )


class FindingScorer:
    """Score a pair of series against 6 interestingness criteria.

    Each criterion returns a float in [0, 1].
    The composite score is a weighted sum of all criteria.
    """

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()
        self.config.validate()

    def score(
        self,
        *,
        correlation: float,
        optimal_lag: int,
        lag_correlation: float,
        granger_p_value: float | None,
        rolling_zscore: float,
        is_new: bool,
    ) -> tuple[list[str], float]:
        """Evaluate all criteria and return (trigger_types, composite_score).

        Parameters
        ----------
        correlation : float
            Pearson r at lag 0.
        optimal_lag : int
            Lag (months) with highest absolute CCF.
        lag_correlation : float
            Correlation at `optimal_lag`.
        granger_p_value : float or None
            Granger p-value (None if test was skipped).
        rolling_zscore : float
            Z-score of recent rolling correlation vs. historical baseline.
        is_new : bool
            True if this pair was not in previous scans.

        Returns
        -------
        (trigger_types, composite_score)
        """
        triggers: list[str] = []
        scores: dict[str, float] = {}

        # 1. High correlation
        s = self._high_correlation(correlation)
        scores["high_correlation"] = s
        if s > 0:
            triggers.append("high_correlation")

        # 2. Newly emerging
        s = self._newly_emerging(is_new)
        scores["newly_emerging"] = s
        if s > 0:
            triggers.append("newly_emerging")

        # 3. Regime change
        s = self._regime_change(rolling_zscore)
        scores["regime_change"] = s
        if s > 0:
            triggers.append("regime_change")

        # 4. Granger causality
        s = self._granger_causality(granger_p_value)
        scores["granger_causality"] = s
        if s > 0:
            triggers.append("granger_causality")

        # 5. Anomalous lag
        s = self._anomalous_lag(optimal_lag, lag_correlation)
        scores["anomalous_lag"] = s
        if s > 0:
            triggers.append("anomalous_lag")

        # 6. Rolling divergence
        s = self._rolling_divergence(rolling_zscore)
        scores["rolling_divergence"] = s
        if s > 0:
            triggers.append("rolling_divergence")

        # Composite weighted score
        composite = sum(
            scores[k] * self.config.weights.get(k, 0)
            for k in scores
        )
        composite = max(0.0, min(1.0, composite))

        return triggers, composite

    # ── individual criteria ───────────────────────────────────────

    def _high_correlation(self, r: float) -> float:
        """Score = |r| if above threshold, else 0."""
        absval = abs(r)
        if absval >= self.config.correlation_threshold:
            return absval
        return 0.0

    def _newly_emerging(self, is_new: bool) -> float:
        return 1.0 if is_new else 0.0

    def _regime_change(self, rolling_zscore: float) -> float:
        """Score = min(|z| / 3, 1) if |z| >= threshold, else 0."""
        absz = abs(rolling_zscore)
        if absz >= self.config.zscore_threshold:
            return min(absz / 3.0, 1.0)
        return 0.0

    def _granger_causality(self, p_value: float | None) -> float:
        """Score = 1 - p if p < threshold, else 0."""
        if p_value is None:
            return 0.0
        if p_value < self.config.granger_p_threshold:
            return 1.0 - p_value
        return 0.0

    def _anomalous_lag(self, lag: int, lag_r: float) -> float:
        """Score based on lag magnitude if lag != 0 and |lag_r| >= threshold."""
        if lag == 0:
            return 0.0
        if abs(lag_r) < self.config.lag_correlation_threshold:
            return 0.0
        # Higher lag = more interesting (capped at 12 months)
        return min(abs(lag) / 12.0, 1.0) * abs(lag_r)

    def _rolling_divergence(self, rolling_zscore: float) -> float:
        """Score = min(|z| / 3, 1) if |z| >= threshold (same logic as regime_change)."""
        absz = abs(rolling_zscore)
        if absz >= self.config.zscore_threshold:
            return min(absz / 3.0, 1.0)
        return 0.0
