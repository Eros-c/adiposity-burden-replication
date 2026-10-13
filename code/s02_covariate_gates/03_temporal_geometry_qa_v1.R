#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

run_dir <- Sys.getenv("FULLCOV_RUN_DIR")
if (!nzchar(run_dir)) stop("FULLCOV_RUN_DIR is required")
future_root <- file.path(.cfg$ROOT, ".codex_tmp", "future_results_v1")
historic_root <- file.path(.cfg$ROOT, ".codex_tmp", "historic_results_v1")
gate_c <- paste0(
  future_root,
  "/02_harmonized_covariates/gate_c_v1_country_coverage_remediated_20260828_122619"
)
base <- file.path(run_dir, "data/0p5deg")
qa_dir <- file.path(run_dir, "qa")

scenarios <- data.table(
  scenario = c("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
  ssp = c("ssp1", "ssp2", "ssp3", "ssp5")
)

template <- rast(file.path(gate_c, "data/0p5deg/country_id_0p5deg.tif"))
pop <- rast(file.path(gate_c, "data/0p5deg/population/population_ssp2_2025_0p5deg.tif"))
hist24 <- rast(file.path(
  historic_root,
  "cache/auxiliary_mrp_0p5deg/covariates/global_covariates_2024_0p5deg.tif"
))
hist20 <- rast(file.path(
  historic_root,
  "cache/auxiliary_mrp_0p5deg/covariates/global_covariates_2020_0p5deg.tif"
))
hist_ntl <- resample(hist24[["ntl_log"]], template, method = "bilinear")
hist_urban <- resample(hist20[["urban_share"]], template, method = "bilinear")
hist_built <- resample(hist20[["built_fraction"]], template, method = "bilinear")
pm_hist <- rast(file.path(base, "pm25_mmr/historical_common/pm25_mmr_ugkg_common_2024_0p5deg.tif"))

weighted_stats <- function(delta, weight) {
  s <- c(delta, weight)
  names(s) <- c("delta", "weight")
  d <- as.data.table(values(s, dataframe = TRUE, na.rm = TRUE))
  d <- d[is.finite(delta) & is.finite(weight) & weight > 0]
  data.table(
    population_weighted_mean_change = weighted.mean(d$delta, d$weight),
    population_weighted_mean_abs_change = weighted.mean(abs(d$delta), d$weight),
    population_weighted_rmse_change = sqrt(weighted.mean(d$delta^2, d$weight)),
    evaluated_cells = nrow(d)
  )
}

transition <- list()
ti <- 0L
for (j in seq_len(nrow(scenarios))) {
  ssp <- scenarios$ssp[j]
  comparisons <- list(
    pm25_mmr_ug_per_kg = list(
      old = pm_hist,
      new = rast(file.path(base, "pm25_mmr/future", sprintf("pm25_mmr_ugkg_%s_2025_0p5deg.tif", ssp)))
    ),
    ntl_log = list(
      old = hist_ntl,
      new = rast(file.path(base, "ntl_log", sprintf("ntl_log_%s_2025_0p5deg.tif", ssp)))
    ),
    urban_share = list(
      old = hist_urban,
      new = rast(file.path(base, "urban_share", sprintf("urban_share_%s_2025_0p5deg.tif", ssp)))
    ),
    built_fraction_proxy = list(
      old = hist_built,
      new = rast(file.path(base, "built_fraction_proxy", sprintf("built_fraction_proxy_%s_2025_0p5deg.tif", ssp)))
    )
  )
  for (variable in names(comparisons)) {
    ti <- ti + 1L
    st <- weighted_stats(comparisons[[variable]]$new - comparisons[[variable]]$old, pop)
    st[, `:=`(variable = variable, scenario = scenarios$scenario[j], from_year = 2024L, to_year = 2025L)]
    transition[[ti]] <- st
  }
}
transition_qa <- rbindlist(transition)
setcolorder(transition_qa, c("variable", "scenario", "from_year", "to_year"))
fwrite(transition_qa, file.path(qa_dir, "Full_covariate_2024_2025_transition_QA_v1.csv"), bom = TRUE)

file_2050 <- function(variable, ssp) {
  switch(
    variable,
    pm25_mmr_ug_per_kg = file.path(base, "pm25_mmr/future", sprintf("pm25_mmr_ugkg_%s_2050_0p5deg.tif", ssp)),
    ntl_log = file.path(base, "ntl_log", sprintf("ntl_log_%s_2050_0p5deg.tif", ssp)),
    urban_share = file.path(base, "urban_share", sprintf("urban_share_%s_2050_0p5deg.tif", ssp)),
    built_fraction_proxy = file.path(base, "built_fraction_proxy", sprintf("built_fraction_proxy_%s_2050_0p5deg.tif", ssp))
  )
}

spread <- list()
si <- 0L
for (variable in c("pm25_mmr_ug_per_kg", "ntl_log", "urban_share", "built_fraction_proxy")) {
  r <- rast(vapply(scenarios$ssp, function(x) file_2050(variable, x), character(1)))
  names(r) <- scenarios$ssp
  range_r <- app(r, function(x) max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  d <- c(range_r, pop)
  names(d) <- c("spread", "weight")
  v <- as.data.table(values(d, dataframe = TRUE, na.rm = TRUE))
  v <- v[is.finite(spread) & is.finite(weight) & weight > 0]
  si <- si + 1L
  spread[[si]] <- data.table(
    variable = variable, year = 2050L, evaluated_cells = nrow(v),
    population_weighted_mean_range = weighted.mean(v$spread, v$weight),
    p50_cell_range = quantile(v$spread, 0.50),
    p95_cell_range = quantile(v$spread, 0.95),
    p99_cell_range = quantile(v$spread, 0.99),
    maximum_cell_range = max(v$spread)
  )
}
fwrite(rbindlist(spread), file.path(qa_dir, "Full_covariate_2050_scenario_spread_QA_v1.csv"), bom = TRUE)

files <- list.files(base, pattern = "[.]tif$", recursive = TRUE, full.names = TRUE)
geometry <- rbindlist(lapply(files, function(path) {
  r <- rast(path)
  ok <- compareGeom(r, template, crs = TRUE, ext = TRUE, rowcol = TRUE, res = TRUE, stopOnError = FALSE)
  data.table(
    path = substring(path, nchar(run_dir) + 2L),
    nrow = nrow(r), ncol = ncol(r), xres = res(r)[1], yres = res(r)[2],
    xmin = ext(r)[1], xmax = ext(r)[2], ymin = ext(r)[3], ymax = ext(r)[4],
    crs = crs(r, proj = TRUE), geometry_matches_primary_grid = ok,
    status = ifelse(ok, "PASS", "FAIL")
  )
}), fill = TRUE)
fwrite(geometry, file.path(qa_dir, "Full_covariate_geometry_QA_v1.csv"), bom = TRUE)
if (any(geometry$status == "FAIL")) stop("Geometry QA failed")

cat("PASS\n")
