"""Real treatment-effect estimation via matched-sample mean difference
and IPTW-weighted mean difference, validated against the simulation's
hidden true ATE -- because that true effect is generated independently
of the adjustment methods, recovery accuracy here is a genuine test."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_ate_matched(matched_df: pd.DataFrame) -> float:
    treated_mean = matched_df.loc[matched_df["treatment"] == 1, "outcome"].mean()
    control_mean = matched_df.loc[matched_df["treatment"] == 0, "outcome"].mean()
    return float(treated_mean - control_mean)


def estimate_ate_iptw(df: pd.DataFrame, weights: pd.Series) -> float:
    treated_mask = df["treatment"] == 1
    control_mask = df["treatment"] == 0
    treated_mean = np.average(df.loc[treated_mask, "outcome"], weights=weights.loc[treated_mask])
    control_mean = np.average(df.loc[control_mask, "outcome"], weights=weights.loc[control_mask])
    return float(treated_mean - control_mean)


def estimate_ate_naive(df: pd.DataFrame) -> float:
    """The confounded, unadjusted estimate -- included to show the
    real bias adjustment removes."""
    treated_mean = df.loc[df["treatment"] == 1, "outcome"].mean()
    control_mean = df.loc[df["treatment"] == 0, "outcome"].mean()
    return float(treated_mean - control_mean)


def validate_ate_recovery(estimates: dict, true_ate: float) -> dict:
    report = {}
    for name, est in estimates.items():
        abs_error = abs(est - true_ate)
        report[name] = {"estimate": round(est, 3), "true_ate": true_ate,
                         "abs_error": round(abs_error, 3), "pct_error": round(100 * abs_error / true_ate, 2)}
    return report
