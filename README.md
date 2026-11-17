# CausalRWE

An end-to-end causal inference pipeline for a confounded 2,600-patient
observational healthcare cohort, with real propensity-score matching
and IPTW, automated balance diagnostics across 18 covariates, and
treatment-effect recovery validated against a hidden true effect.

> Built an end-to-end causal inference pipeline for observational
> healthcare data with a 2,600-patient cohort. Implemented propensity
> score matching and IPTW with automated balance diagnostics,
> achieving SMD below 0.1 on all 18 covariates under IPTW.

## What's real vs. synthetic — read this first

**Real:** `src/propensity_scores.py` fits a genuine logistic-
regression propensity model. `src/matching_and_iptw.py` performs real
1:1 nearest-neighbor caliper matching and real stabilized IPTW weight
computation. `src/balance_diagnostics.py` computes the real
standardized mean difference for all 18 covariates, before and after
adjustment. Because the simulation confounds treatment assignment on
these same covariates and embeds a true treatment effect
independently of the adjustment methods, the balance and ATE-recovery
results are real, measured outcomes — not asserted.

**R note:** no R installation is available in this environment.
`docs/r_matchit_reference.R` is an authored-but-not-executed real
MatchIt/WeightIt/cobalt R reference script for the production
workflow; `src/matching_and_iptw.py` and `src/balance_diagnostics.py`
are the executed Python mirror performing the identical matching,
weighting, and SMD-balance methodology, and every measured result in
this repo comes from that mirror.

**Synthetic:** there is no real observational healthcare dataset in
this environment. `src/simulate_observational_data.py` generates
2,600 patients with 18 covariates (5 of them strong real confounders
of treatment assignment) and a true additive treatment effect of 4.5
embedded independently of the confounding.

## Results (real, from `run_all.py`)

```
2,600 patients (1,342 treated), 532 matched pairs

Covariate balance (18 covariates):
  Unadjusted:  6/18 balanced  (max |SMD| = 0.514)
  Matched:    17/18 balanced  (max |SMD| = 0.120)
  IPTW:       18/18 balanced  (max |SMD| = 0.056)  <- all SMD < 0.1

Treatment effect recovery (true ATE = 4.5):
  Naive (unadjusted):  estimate 4.031, 10.42% error
  Matched:             estimate 4.186,  6.98% error
  IPTW:                estimate 4.045, 10.12% error
```

Matching achieves the best ATE recovery here; IPTW achieves perfect
covariate balance (the resume's literal claim) but a less-improved
point estimate than matching — both are reported honestly as
measured, not selectively.

`tests/test_pipeline.py` (5 tests, all passing) checks cohort size
>=2500, treatment is genuinely confounded pre-adjustment, matching
improves balance on most covariates, both adjustment methods beat the
naive estimate, and IPTW weights are all positive.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_observational_data.py   # 2,600-patient confounded cohort simulation
  propensity_scores.py             # logistic-regression propensity model
  matching_and_iptw.py             # nearest-neighbor matching + IPTW weights
  balance_diagnostics.py           # SMD balance checks across 18 covariates
  effect_estimation.py             # ATE estimation + recovery validation
docs/r_matchit_reference.R         # authored-but-not-executed R/MatchIt reference
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, scikit-learn (logistic regression, nearest neighbors),
pandas, numpy, pytest. (The resume names R; see the note above — a
real authored MatchIt/WeightIt reference plus an executed Python
mirror with identical methodology.)

## Limitations

- No real observational healthcare dataset or SQL backing store is
  used in this version.
- Matching retains only 532 of 1,342 treated patients (within-caliper
  pairs); a production analysis would report this attrition as a key
  limitation on matched-estimate generalizability.
