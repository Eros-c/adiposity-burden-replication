#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

run_dir <- Sys.getenv("FULLCOV_RUN_DIR")
if (!nzchar(run_dir)) stop("FULLCOV_RUN_DIR is required")
future_root <- file.path(.cfg$ROOT, ".codex_tmp", "future_results_v1")
template <- rast(file.path(
  future_root,
  "02_harmonized_covariates/gate_c_v1_country_coverage_remediated_20260828_122619/data/0p5deg/country_id_0p5deg.tif"
))
base <- file.path(run_dir, "data/0p5deg")
files <- list.files(base, pattern = "[.]tif$", recursive = TRUE, full.names = TRUE)

geometry <- rbindlist(lapply(files, function(path) {
  r <- rast(path)
  ok <- compareGeom(r, template, crs = TRUE, ext = TRUE, rowcol = TRUE,
                    res = TRUE, stopOnError = FALSE)
  data.table(
    path = substring(path, nchar(run_dir) + 2L),
    nrow = nrow(r), ncol = ncol(r), xres = res(r)[1], yres = res(r)[2],
    xmin = ext(r)[1], xmax = ext(r)[2], ymin = ext(r)[3], ymax = ext(r)[4],
    crs = crs(r, proj = TRUE), geometry_matches_primary_grid = ok,
    status = ifelse(ok, "PASS", "FAIL")
  )
}), fill = TRUE)
fwrite(geometry, file.path(run_dir, "qa/Full_covariate_geometry_QA_v2.csv"), bom = TRUE)

count <- data.table(
  variable = c("pm25_historical", "pm25_future", "ntl_future",
               "urban_share_future", "built_proxy_future"),
  expected = c(19L, 104L, 104L, 104L, 104L),
  observed = c(
    length(list.files(file.path(base, "pm25_mmr_trailing5y_v2/historical_common"), pattern = "[.]tif$")),
    length(list.files(file.path(base, "pm25_mmr_trailing5y_v2/future"), pattern = "[.]tif$")),
    length(list.files(file.path(base, "ntl_log"), pattern = "[.]tif$")),
    length(list.files(file.path(base, "urban_share"), pattern = "[.]tif$")),
    length(list.files(file.path(base, "built_fraction_proxy"), pattern = "[.]tif$"))
  )
)
count[, status := ifelse(expected == observed, "PASS", "FAIL")]
fwrite(count, file.path(run_dir, "qa/Full_covariate_expected_file_count_QA_v2.csv"), bom = TRUE)

if (length(files) != 435L || any(geometry$status == "FAIL") || any(count$status == "FAIL")) {
  stop("Final geometry/file-count QA failed")
}
cat("PASS: 435 final rasters\n")
