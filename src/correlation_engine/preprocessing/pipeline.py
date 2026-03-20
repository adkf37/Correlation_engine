"""Configurable preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from correlation_engine.preprocessing.align import align_frequencies
from correlation_engine.preprocessing.missing import handle_missing, report_missing
from correlation_engine.preprocessing.transform import (
    check_stationarity_all,
    make_stationary,
)


class PreprocessingPipeline:
    """Chain preprocessing steps and run them in order.

    Parameters
    ----------
    steps : list of (name, params) tuples
        Each step is a tuple of (step_name, kwargs_dict).
        Supported step names: 'align', 'missing', 'transform'.

    Example
    -------
    >>> pipeline = PreprocessingPipeline([
    ...     ('align', {'target_freq': 'M'}),
    ...     ('missing', {'strategy': 'interpolate'}),
    ...     ('transform', {'method': 'diff'}),
    ... ])
    >>> clean_df = pipeline.run(raw_df)
    >>> pipeline.report()
    """

    _STEP_NAMES = {"align", "missing", "transform"}

    def __init__(self, steps: list[tuple[str, dict]] | None = None):
        self.steps = steps or []
        for name, _ in self.steps:
            if name not in self._STEP_NAMES:
                raise ValueError(
                    f"Unknown step '{name}'. Choose from: {self._STEP_NAMES}"
                )
        self._report: dict = {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all steps in order and return the processed DataFrame."""
        self._report = {}
        result = df.copy()

        for step_name, params in self.steps:
            rows_before = len(result)
            cols_before = list(result.columns)

            if step_name == "align":
                result = align_frequencies(result, **params)
                self._report["align"] = {
                    "rows_before": rows_before,
                    "rows_after": len(result),
                    "params": params,
                }

            elif step_name == "missing":
                missing_before = report_missing(result)
                result = handle_missing(result, **params)
                cols_after = list(result.columns)
                self._report["missing"] = {
                    "missing_before": missing_before.to_dict(),
                    "rows_before": rows_before,
                    "rows_after": len(result),
                    "columns_dropped": [c for c in cols_before if c not in cols_after],
                    "params": params,
                }

            elif step_name == "transform":
                stationarity_before = check_stationarity_all(result)
                result, transform_report = make_stationary(result, **params)
                stationarity_after = check_stationarity_all(result)
                self._report["transform"] = {
                    "stationarity_before": stationarity_before.to_dict(),
                    "stationarity_after": stationarity_after.to_dict(),
                    "transforms_applied": transform_report,
                    "rows_before": rows_before,
                    "rows_after": len(result),
                    "params": params,
                }

        return result

    def report(self) -> dict:
        """Return a summary of what each step did during the last run."""
        return self._report
