# Compare Pathfinder pilot draws against the production mean-field VB fit.
# Reads cmdstan pathfinder CSV output and the rstan MFVB summary, aligns
# parameter names, and reports posterior-mean agreement and SD inflation.
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
pilot_root <- if (length(args) >= 1) args[[1]] else
  file.path(.cfg$RESULTS, "04_future_capable_mrp", "full_covariate_mrp_refit_v1_20260905", "pathfinder_pilot_v1")
refit_root <- file.path(
  dirname(dirname(pilot_root)),
  "full_covariate_mrp_refit_v1_20260905"
)
mfvb_summary_path <- file.path(
  refit_root, "model_fit/models/full_model_parameter_summary_v1.csv"
)
draws_path <- file.path(pilot_root, "models/pathfinder_draws_v1.csv")

cmdstan_to_rstan_name <- function(column) {
  # "beta.1.1" -> "beta[1,1]", "sigma_country_intercept.3" -> "sigma_country_intercept[3]"
  if (!grepl("^beta\\.", column) && !grepl("^sigma_", column)) return(column)
  base <- sub("\\.[0-9].*$", "", column)
  if (identical(base, column)) return(column)
  indices <- gsub("\\.", ",", sub(paste0("^", base, "\\."), "", column))
  paste0(base, "[", indices, "]")
}

draws <- fread(draws_path, showProgress = FALSE)
draws <- draws[, setdiff(names(draws), c("lp__", "lp_approx__")), with = FALSE]

long <- data.table(
  parameter = unname(vapply(names(draws), cmdstan_to_rstan_name, character(1))),
  mean = vapply(draws, mean, numeric(1)),
  sd = vapply(draws, sd, numeric(1)),
  q2_5 = vapply(draws, quantile, numeric(1), probs = 0.025, names = FALSE),
  q97_5 = vapply(draws, quantile, numeric(1), probs = 0.975, names = FALSE)
)

mfvb <- fread(mfvb_summary_path)
setnames(mfvb, c("parameter", "mean_mfvb", "se_mean_mfvb", "sd_mfvb",
                 "q2_5_mfvb", "q25_mfvb", "q50_mfvb", "q75_mfvb",
                 "q97_5_mfvb", "n_eff_mfvb", "Rhat_mfvb")[seq_len(ncol(mfvb))])

comparison <- merge(
  long, mfvb[, .(parameter, mean_mfvb, sd_mfvb)], by = "parameter", all = FALSE
)
comparison[, `:=`(
  mean_abs_diff = abs(mean - mean_mfvb),
  sd_ratio_pathfinder_over_mfvb = sd / pmax(sd_mfvb, 1e-8)
)]
setorder(comparison, parameter)

dir.create(file.path(pilot_root, "qa"), showWarnings = FALSE)
dir.create(file.path(pilot_root, "reports"), showWarnings = FALSE)
fwrite(comparison, file.path(pilot_root, "qa/pathfinder_vs_meanfield_summary_v1.csv"))

beta_rows <- comparison[grepl("^beta\\[", parameter)]
sigma_rows <- comparison[grepl("^sigma_", parameter)]

elapsed <- tryCatch(
  fread(file.path(pilot_root, "qa/pathfinder_elapsed_v1.csv")),
  error = function(e) NULL
)

diagnostics <- list(
  status = "complete",
  n_parameters_compared = nrow(comparison),
  n_draws = nrow(draws),
  elapsed_seconds = if (is.null(elapsed)) NA_real_ else elapsed$V2[1],
  beta_mean_correlation = unname(cor(beta_rows$mean, beta_rows$mean_mfvb)),
  beta_max_abs_mean_diff = beta_rows[, max(mean_abs_diff)],
  beta_sd_ratio_median = beta_rows[, median(sd_ratio_pathfinder_over_mfvb)],
  beta_sd_ratio_p05 = beta_rows[, quantile(sd_ratio_pathfinder_over_mfvb, 0.05)],
  beta_sd_ratio_p95 = beta_rows[, quantile(sd_ratio_pathfinder_over_mfvb, 0.95)],
  sigma_sd_ratio = as.list(setNames(
    sigma_rows$sd_ratio_pathfinder_over_mfvb, sigma_rows$parameter
  )),
  compared_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
write_json(diagnostics, file.path(pilot_root, "qa/pathfinder_diagnostics_v1.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA)

report <- c(
  "# Pathfinder vs mean-field VB comparison",
  "",
  sprintf("- draws analysed: %d", nrow(draws)),
  sprintf("- parameters compared: %d", nrow(comparison)),
  if (!is.null(elapsed)) sprintf("- pathfinder elapsed: %.0f s", elapsed$V2[1]) else NULL,
  sprintf("- beta posterior-mean correlation (pathfinder vs MFVB): %.4f",
          diagnostics$beta_mean_correlation),
  sprintf("- beta max |mean difference|: %.4f", diagnostics$beta_max_abs_mean_diff),
  sprintf("- beta SD ratio (pathfinder/MFVB): median %.3f, 90%% range %.3f-%.3f",
          diagnostics$beta_sd_ratio_median,
          diagnostics$beta_sd_ratio_p05, diagnostics$beta_sd_ratio_p95),
  "",
  "Interpretation: a median SD ratio above 1 indicates that the mean-field",
  "approximation understates posterior uncertainty; ratios near 1 mean the",
  "MFVB posterior spread was already adequate for these parameters."
)
writeLines(report, file.path(pilot_root, "reports/pathfinder_report_v1.md"))
cat(paste(report, collapse = "\n"), "\n")
