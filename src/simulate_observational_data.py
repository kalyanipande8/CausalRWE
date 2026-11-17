"""Simulate a confounded observational healthcare cohort of 2,600
patients with 18 covariates, where treatment assignment is genuinely
confounded (sicker/older patients more likely treated) and a known
true causal treatment effect drives the outcome independently of the
confounding -- the ground truth a causal inference pipeline must
recover despite the confounding."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 2600
N_COVARIATES = 18
SEED = 907
TRUE_ATE = 4.5


def simulate(n: int = N_PATIENTS, n_covariates: int = N_COVARIATES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    cov_names = [f"cov_{i:02d}" for i in range(n_covariates)]

    covariates = rng.normal(0, 1, (n, n_covariates))
    confounder_weights = rng.normal(0, 0.25, n_covariates)
    confounder_weights[:5] *= 1.5  # first 5 covariates are real confounders

    propensity_logit = covariates @ confounder_weights
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)

    outcome_weights = rng.normal(0, 1.0, n_covariates)
    baseline_outcome = 50 + covariates @ outcome_weights
    outcome = baseline_outcome + TRUE_ATE * treatment + rng.normal(0, 3, n)

    df = pd.DataFrame(covariates, columns=cov_names)
    df.insert(0, "patient_id", [f"PT-{i:05d}" for i in range(n)])
    df["treatment"] = treatment
    df["outcome"] = outcome
    return df, cov_names, TRUE_ATE


if __name__ == "__main__":
    df, cov_names, true_ate = simulate()
    print(df.shape, df["treatment"].mean(), "true ATE:", true_ate)
