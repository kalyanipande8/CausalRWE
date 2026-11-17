"""Real automated covariate-balance diagnostics: computes the
standardized mean difference (SMD) between treated and control groups
for all 18 covariates, before and after adjustment -- the standard
causal-inference balance check, with the real SMD < 0.1 threshold
this account's resume bullet cites."""
from __future__ import annotations

import numpy as np
import pandas as pd

SMD_THRESHOLD = 0.1


def standardized_mean_diff(treated: np.ndarray, control: np.ndarray) -> float:
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return (treated.mean() - control.mean()) / pooled_sd if pooled_sd > 0 else 0.0


def compute_balance(df: pd.DataFrame, cov_names: list, weights: pd.Series = None) -> pd.DataFrame:
    results = []
    for cov in cov_names:
        treated_vals = df.loc[df["treatment"] == 1, cov].values
        control_vals = df.loc[df["treatment"] == 0, cov].values

        if weights is not None:
            w_treated = weights.loc[df["treatment"] == 1].values
            w_control = weights.loc[df["treatment"] == 0].values
            treated_mean = np.average(treated_vals, weights=w_treated)
            control_mean = np.average(control_vals, weights=w_control)
            pooled_sd = np.sqrt((treated_vals.var(ddof=1) + control_vals.var(ddof=1)) / 2)
            smd = (treated_mean - control_mean) / pooled_sd if pooled_sd > 0 else 0.0
        else:
            smd = standardized_mean_diff(treated_vals, control_vals)

        results.append({"covariate": cov, "smd": round(float(smd), 4), "balanced": abs(smd) < SMD_THRESHOLD})

    return pd.DataFrame(results)


def balance_summary(balance_df: pd.DataFrame) -> dict:
    return {"n_covariates": len(balance_df), "n_balanced": int(balance_df["balanced"].sum()),
            "max_abs_smd": round(float(balance_df["smd"].abs().max()), 4),
            "all_balanced": bool(balance_df["balanced"].all())}
