# Authored-but-not-executed reference: the real R (MatchIt/WeightIt)
# causal inference workflow this pipeline would run in production. No
# R installation is available in this environment, so
# src/matching_and_iptw.py and src/balance_diagnostics.py implement
# the identical nearest-neighbor matching, IPTW, and SMD balance
# diagnostics in Python as the executed mirror, and all measured
# results in this repo come from that Python mirror.

library(MatchIt)
library(WeightIt)
library(cobalt)

cohort <- read.csv("data/observational_cohort.csv")
covariates <- paste0("cov_", sprintf("%02d", 0:17))
formula <- as.formula(paste("treatment ~", paste(covariates, collapse = " + ")))

# 1:1 nearest-neighbor propensity-score matching
match_out <- matchit(formula, data = cohort, method = "nearest", distance = "glm", caliper = 0.05)
matched_data <- match.data(match_out)

# IPTW with stabilized weights
weight_out <- weightit(formula, data = cohort, method = "glm", estimand = "ATE", stabilize = TRUE)
cohort$iptw_weight <- weight_out$weights

# Balance diagnostics (the same SMD computation as
# src/balance_diagnostics.py's compute_balance())
bal_tab <- bal.tab(match_out, un = TRUE, m.threshold = 0.1)
print(bal_tab)

# Treatment effect estimation
ate_matched <- mean(matched_data$outcome[matched_data$treatment == 1]) -
  mean(matched_data$outcome[matched_data$treatment == 0])

ate_iptw <- with(cohort, weighted.mean(outcome[treatment == 1], iptw_weight[treatment == 1]) -
  weighted.mean(outcome[treatment == 0], iptw_weight[treatment == 0]))

cat(sprintf("Matched ATE: %.3f, IPTW ATE: %.3f\n", ate_matched, ate_iptw))
