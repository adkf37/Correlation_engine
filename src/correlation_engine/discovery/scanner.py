"""DiscoveryScanner — exhaustive pairwise analysis across the watchlist."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from correlation_engine.analysis.correlation import compute_correlation_matrix
from correlation_engine.analysis.granger import granger_causality_test
from correlation_engine.analysis.lag import compute_cross_correlation
from correlation_engine.analysis.rolling import compute_rolling_correlation
from correlation_engine.discovery.findings import Finding
from correlation_engine.discovery.scoring import FindingScorer, ScoringConfig

logger = logging.getLogger(__name__)


class ScanConfig:
    """Runtime configuration for a single scan run."""

    def __init__(
        self,
        *,
        rolling_window: int = 12,
        min_r_for_granger: float = 0.3,
        min_score_threshold: float = 0.0,
        max_lag: int = 12,
        scoring: ScoringConfig | None = None,
        name_map: dict[str, str] | None = None,
    ):
        self.rolling_window = rolling_window
        self.min_r_for_granger = min_r_for_granger
        self.min_score_threshold = min_score_threshold
        self.max_lag = max_lag
        self.scoring = scoring or ScoringConfig()
        self.name_map = name_map or {}


class DiscoveryScanner:
    """Scan all pairwise combinations and produce ranked Finding records.

    Parameters
    ----------
    config : ScanConfig
        Thresholds, windows, pre-filter settings.
    """

    def __init__(self, config: ScanConfig | None = None):
        self.config = config or ScanConfig()
        self._scorer = FindingScorer(self.config.scoring)

    def scan(
        self,
        series_dict: dict[str, pd.Series],
        was_seen_fn: Callable[[str, str], bool] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Finding]:
        """Run exhaustive pairwise analysis.

        Parameters
        ----------
        series_dict : dict
            Mapping of series_id → pd.Series (preprocessed, monthly).
        was_seen_fn : callable, optional
            ``(series_a, series_b) -> bool`` returning True if pair appeared
            in previous scans. Used for ``is_new`` scoring.
        on_progress : callable, optional
            ``(completed, total)`` callback for progress tracking.

        Returns
        -------
        list[Finding]
            Sorted by ``interestingness_score`` descending.
        """
        scan_id = str(uuid.uuid4())
        scanned_at = datetime.now(timezone.utc).isoformat()

        keys = sorted(series_dict.keys())
        pairs = list(combinations(keys, 2))
        total = len(pairs)
        logger.info("Scanning %d pairs from %d series.", total, len(keys))

        # Pre-compute full correlation matrix for fast r lookup
        df = pd.DataFrame(series_dict)
        corr_matrix = compute_correlation_matrix(df, method="pearson")

        findings: list[Finding] = []

        for idx, (a, b) in enumerate(pairs):
            if on_progress and idx % 50 == 0:
                on_progress(idx, total)

            finding = self._analyze_pair(
                a, b,
                series_dict[a], series_dict[b],
                corr_matrix,
                scan_id, scanned_at,
                was_seen_fn,
            )
            if finding is not None:
                findings.append(finding)

        if on_progress:
            on_progress(total, total)

        findings.sort(key=lambda f: f.interestingness_score, reverse=True)
        logger.info(
            "Scan complete: %d findings from %d pairs.", len(findings), total,
        )
        return findings

    # ── private ───────────────────────────────────────────────────

    def _analyze_pair(
        self,
        a: str,
        b: str,
        sa: pd.Series,
        sb: pd.Series,
        corr_matrix: pd.DataFrame,
        scan_id: str,
        scanned_at: str,
        was_seen_fn: Callable[[str, str], bool] | None,
    ) -> Finding | None:
        """Analyze a single pair and return a Finding if any trigger fires."""
        cfg = self.config
        name_map = cfg.name_map

        # 1. Pearson r at lag 0 (from pre-computed matrix)
        r = corr_matrix.loc[a, b]
        if np.isnan(r):
            return None

        # 2. Cross-correlation → optimal lag
        try:
            ccf_df = compute_cross_correlation(sa, sb, max_lag=cfg.max_lag)
            best_idx = ccf_df["correlation"].abs().idxmax()
            optimal_lag = int(ccf_df.loc[best_idx, "lag"])
            lag_corr = float(ccf_df.loc[best_idx, "correlation"])
        except Exception:
            optimal_lag = 0
            lag_corr = float(r)

        # 3. Rolling correlation → z-score
        rolling_zscore = self._compute_rolling_zscore(sa, sb, cfg.rolling_window)

        # 4. Granger causality (pre-filter: skip if |r| too low)
        granger_p: float | None = None
        granger_dir: str | None = None
        if abs(r) >= cfg.min_r_for_granger:
            granger_p, granger_dir = self._run_granger(sa, sb, a, b)

        # 5. Check if pair is new
        is_new = True
        if was_seen_fn is not None:
            is_new = not was_seen_fn(a, b)

        # 6. Score
        triggers, score = self._scorer.score(
            correlation=r,
            optimal_lag=optimal_lag,
            lag_correlation=lag_corr,
            granger_p_value=granger_p,
            rolling_zscore=rolling_zscore,
            is_new=is_new,
        )

        # Filter: must have at least one trigger or meet score threshold
        if not triggers and score < cfg.min_score_threshold:
            return None

        return Finding(
            scan_id=scan_id,
            scanned_at=scanned_at,
            series_a=a,
            series_b=b,
            series_a_name=name_map.get(a, a),
            series_b_name=name_map.get(b, b),
            correlation=float(r),
            optimal_lag=optimal_lag,
            lag_correlation=lag_corr,
            granger_p_value=granger_p,
            granger_direction=granger_dir,
            rolling_zscore=rolling_zscore,
            regime_change_detected=abs(rolling_zscore) >= cfg.scoring.zscore_threshold,
            trigger_types=triggers,
            interestingness_score=score,
            is_new=is_new,
            frequency="M",
        )

    def _compute_rolling_zscore(
        self,
        sa: pd.Series,
        sb: pd.Series,
        window: int,
    ) -> float:
        """Z-score of the most recent rolling correlation vs. full-history baseline."""
        try:
            rolling = compute_rolling_correlation(sa, sb, window=window)
            rolling_clean = rolling.dropna()
            if len(rolling_clean) < 6:
                return 0.0
            current = rolling_clean.iloc[-1]
            mean = rolling_clean.mean()
            std = rolling_clean.std()
            if std < 1e-10:
                return 0.0
            return float((current - mean) / std)
        except Exception:
            return 0.0

    def _run_granger(
        self,
        sa: pd.Series,
        sb: pd.Series,
        a: str,
        b: str,
    ) -> tuple[float | None, str | None]:
        """Run bidirectional Granger test and return (best_p, direction)."""
        pair_df = pd.DataFrame({a: sa, b: sb}).dropna()
        if len(pair_df) < 25:
            return None, None

        try:
            res_ab = granger_causality_test(pair_df, target=b, predictor=a, max_lag=6)
            res_ba = granger_causality_test(pair_df, target=a, predictor=b, max_lag=6)
        except Exception:
            return None, None

        p_ab = res_ab["p_value"]
        p_ba = res_ba["p_value"]

        sig = self.config.scoring.granger_p_threshold
        a_causes_b = p_ab < sig
        b_causes_a = p_ba < sig

        if a_causes_b and b_causes_a:
            direction = "bidirectional"
            best_p = min(p_ab, p_ba)
        elif a_causes_b:
            direction = "a_causes_b"
            best_p = p_ab
        elif b_causes_a:
            direction = "b_causes_a"
            best_p = p_ba
        else:
            direction = None
            best_p = min(p_ab, p_ba)

        return best_p, direction
