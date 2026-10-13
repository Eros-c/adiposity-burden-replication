#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

run_dir <- Sys.getenv("FULLCOV_RUN_DIR")
if (!nzchar(run_dir)) stop("FULLCOV_RUN_DIR is required")
future_root <- file.path(.cfg$ROOT, ".codex_tmp", "future_results_v1")
gate_c <- paste0(
  future_root,
  "/02_harmonized_covariates/gate_c_v1_country_coverage_remediated_20260828_122619"
)
base <- file.path(run_dir, "data/0p5deg/pm25_mmr_trailing5y_v2")
pop <- rast(file.path(gate_c, "data/0p5deg/population/population_ssp2_2025_0p5deg.tif"))
hist <- rast(file.path(base, "historical_common/pm25_mmr_ugkg_trailing5y_common_2024_0p5deg.tif"))
scenarios <- data.table(
  scenario = c("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
  ssp = c("ssp1", "ssp2", "ssp3", "ssp5")
)

rows <- lapply(seq_len(nrow(scenarios)), function(i) {
  fut <- rast(file.path(
    base, "future",
    sprintf("pm25_mmr_ugkg_trailing5y_%s_2025_0p5deg.tif", scenarios$ssp[i])
  ))
  s <- c(fut - hist, pop)
  names(s) <- c("delta", "weight")
  d <- as.data.table(values(s, dataframe = TRUE, na.rm = TRUE))
  d <- d[is.finite(delta) & is.finite(weight) & weight > 0]
  data.table(
    variable = "pm25_mmr_ug_per_kg_trailing5y",
    scenario = scenarios$scenario[i], from_year = 2024L, to_year = 2025L,
    population_weighted_mean_change = weighted.mean(d$delta, d$weight),
    population_weighted_mean_abs_change = weighted.mean(abs(d$delta), d$weight),
    population_weighted_rmse_change = sqrt(weighted.mean(d$delta^2, d$weight)),
    evaluated_cells = nrow(d)
  )
})
fwrite(
  rbindlist(rows),
  file.path(run_dir, "qa/PM25_trailing5y_2024_2025_transition_QA_v2.csv"),
  bom = TRUE
)
cat("PASS\n")
