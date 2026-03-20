"""Analysis: correlation matrix, lag analysis, rolling, significance, Granger, DCC-GARCH."""

from correlation_engine.analysis.correlation import (
    cluster_correlation_matrix,
    compute_correlation_matrix,
)
from correlation_engine.analysis.dcc_garch import DccResult, fit_dcc_garch
from correlation_engine.analysis.granger import (
    granger_causality_matrix,
    granger_causality_test,
    granger_summary,
)
from correlation_engine.analysis.lag import (
    compute_cross_correlation,
    compute_lead_lag_matrix,
)
from correlation_engine.analysis.rolling import (
    compute_rolling_correlation,
    compute_rolling_matrix,
    window_sensitivity,
)
from correlation_engine.analysis.significance import (
    adjust_pvalues,
    bootstrap_correlation_ci,
    bootstrap_correlation_matrix_ci,
    compute_pvalue_matrix,
    flag_significant,
)

__all__ = [
    "compute_correlation_matrix",
    "cluster_correlation_matrix",
    "compute_cross_correlation",
    "compute_lead_lag_matrix",
    "compute_rolling_correlation",
    "compute_rolling_matrix",
    "window_sensitivity",
    "bootstrap_correlation_ci",
    "bootstrap_correlation_matrix_ci",
    "compute_pvalue_matrix",
    "adjust_pvalues",
    "flag_significant",
    "granger_causality_test",
    "granger_causality_matrix",
    "granger_summary",
    "fit_dcc_garch",
    "DccResult",
]
