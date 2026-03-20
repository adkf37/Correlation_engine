"""Preprocessing: frequency alignment, missing data, stationarity transforms."""

from correlation_engine.preprocessing.align import align_frequencies
from correlation_engine.preprocessing.missing import handle_missing, report_missing
from correlation_engine.preprocessing.pipeline import PreprocessingPipeline
from correlation_engine.preprocessing.transform import (
    check_stationarity,
    check_stationarity_all,
    make_stationary,
)

__all__ = [
    "align_frequencies",
    "handle_missing",
    "report_missing",
    "check_stationarity",
    "check_stationarity_all",
    "make_stationary",
    "PreprocessingPipeline",
]
