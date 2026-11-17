"""Real propensity-score matching (1:1 nearest-neighbor on the
estimated propensity score) and real inverse-probability-of-treatment
weighting (IPTW, stabilized weights) -- the two standard causal
adjustment methods this pipeline compares."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_match(df: pd.DataFrame, propensity: pd.Series, caliper: float = 0.1) -> pd.DataFrame:
    treated = df[df["treatment"] == 1].copy()
    control = df[df["treatment"] == 0].copy()

    treated_ps = propensity.loc[treated.index].values.reshape(-1, 1)
    control_ps = propensity.loc[control.index].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1).fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)

    matched_pairs = []
    used_controls = set()
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper and idx not in used_controls:
            matched_pairs.append((treated.index[i], control.index[idx]))
            used_controls.add(idx)

    matched_treated_idx = [p[0] for p in matched_pairs]
    matched_control_idx = [p[1] for p in matched_pairs]
    return df.loc[matched_treated_idx + matched_control_idx]


def compute_iptw_weights(df: pd.DataFrame, propensity: pd.Series) -> pd.Series:
    treated_mean = df["treatment"].mean()
    weights = np.where(df["treatment"] == 1, treated_mean / propensity, (1 - treated_mean) / (1 - propensity))
    return pd.Series(weights, index=df.index, name="iptw_weight")
