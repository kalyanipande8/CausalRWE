"""Real propensity-score estimation via logistic regression on the
18 observed covariates -- the standard first step of a causal
inference pipeline for observational data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity_scores(df: pd.DataFrame, cov_names: list) -> pd.Series:
    X = df[cov_names].values
    y = df["treatment"].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    scores = np.clip(scores, 0.02, 0.98)  # avoid extreme-weight instability in IPTW
    return pd.Series(scores, index=df.index, name="propensity_score")
