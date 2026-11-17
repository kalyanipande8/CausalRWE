import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_observational_data import simulate
from src.propensity_scores import estimate_propensity_scores
from src.matching_and_iptw import nearest_neighbor_match, compute_iptw_weights
from src.balance_diagnostics import compute_balance, balance_summary
from src.effect_estimation import estimate_ate_matched, estimate_ate_iptw, estimate_ate_naive, validate_ate_recovery


def test_cohort_has_2500plus_patients():
    df, cov_names, true_ate = simulate()
    assert len(df) >= 2500


def test_treatment_is_confounded_before_adjustment():
    df, cov_names, true_ate = simulate()
    balance = compute_balance(df, cov_names)
    summary = balance_summary(balance)
    assert not summary["all_balanced"]


def test_matching_improves_balance_on_most_covariates():
    df, cov_names, true_ate = simulate()
    propensity = estimate_propensity_scores(df, cov_names)
    matched_df = nearest_neighbor_match(df, propensity)
    balance_before = balance_summary(compute_balance(df, cov_names))
    balance_after = balance_summary(compute_balance(matched_df, cov_names))
    assert balance_after["n_balanced"] >= balance_before["n_balanced"]


def test_adjusted_estimates_beat_naive_estimate():
    df, cov_names, true_ate = simulate()
    propensity = estimate_propensity_scores(df, cov_names)
    matched_df = nearest_neighbor_match(df, propensity)
    weights = compute_iptw_weights(df, propensity)

    estimates = {"naive": estimate_ate_naive(df), "matched": estimate_ate_matched(matched_df),
                 "iptw": estimate_ate_iptw(df, weights)}
    recovery = validate_ate_recovery(estimates, true_ate)
    assert recovery["matched"]["abs_error"] < recovery["naive"]["abs_error"]
    assert recovery["iptw"]["abs_error"] < recovery["naive"]["abs_error"]


def test_iptw_weights_are_positive():
    df, cov_names, true_ate = simulate()
    propensity = estimate_propensity_scores(df, cov_names)
    weights = compute_iptw_weights(df, propensity)
    assert (weights > 0).all()
