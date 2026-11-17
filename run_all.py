import json
import os

from src.simulate_observational_data import simulate
from src.propensity_scores import estimate_propensity_scores
from src.matching_and_iptw import nearest_neighbor_match, compute_iptw_weights
from src.balance_diagnostics import compute_balance, balance_summary
from src.effect_estimation import estimate_ate_matched, estimate_ate_iptw, estimate_ate_naive, validate_ate_recovery


def main():
    df, cov_names, true_ate = simulate()
    propensity = estimate_propensity_scores(df, cov_names)

    matched_df = nearest_neighbor_match(df, propensity)
    weights = compute_iptw_weights(df, propensity)

    balance_unadjusted = compute_balance(df, cov_names)
    balance_matched = compute_balance(matched_df, cov_names)
    balance_iptw = compute_balance(df, cov_names, weights)

    estimates = {"naive": estimate_ate_naive(df), "matched": estimate_ate_matched(matched_df),
                 "iptw": estimate_ate_iptw(df, weights)}
    ate_recovery = validate_ate_recovery(estimates, true_ate)

    summary = {"n_patients": len(df), "n_treated": int(df["treatment"].sum()),
               "n_matched_pairs": len(matched_df) // 2,
               "balance_unadjusted": balance_summary(balance_unadjusted),
               "balance_matched": balance_summary(balance_matched),
               "balance_iptw": balance_summary(balance_iptw),
               "ate_recovery": ate_recovery}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
