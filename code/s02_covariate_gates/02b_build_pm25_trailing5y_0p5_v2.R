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
template <- rast(file.path(gate_c, "data/0p5deg/country_id_0p5deg.tif"))
template[] <- NA
crs(template) <- "EPSG:4326"

native <- file.path(run_dir, "data/pm25_annual_native_trailing5y_v2")
hist_dir <- file.path(run_dir, "data/0p5deg/pm25_mmr_trailing5y_v2/historical_common")
future_dir <- file.path(run_dir, "data/0p5deg/pm25_mmr_trailing5y_v2/future")
qa_dir <- file.path(run_dir, "qa")
dir.create(hist_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(future_dir, recursive = TRUE, showWarnings = FALSE)

write_new <- function(r, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path)
  writeRaster(
    r, path, overwrite = FALSE,
    wopt = list(
      datatype = "FLT4S", NAflag = -9999,
      gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES")
    )
  )
}

qa <- list(); qi <- 0L
record <- function(variable, scenario, year, r) {
  g <- global(r, c("min", "max", "mean"), na.rm = TRUE)
  qi <<- qi + 1L
  qa[[qi]] <<- data.table(
    variable = variable, scenario = scenario, year = year,
    min = g[1, "min"], max = g[1, "max"], mean = g[1, "mean"]
  )
}

hist <- rast(file.path(native, "pm25_mmr_ugkg_trailing5y_historical_common_1990_2024.nc"))
crs(hist) <- "EPSG:4326"
hist05 <- project(hist, template, method = "bilinear")
for (year in 2006:2024) {
  r <- hist05[[year - 1990 + 1L]]
  names(r) <- "pm25_mmr_ug_per_kg_trailing5y"
  write_new(r, file.path(hist_dir, sprintf("pm25_mmr_ugkg_trailing5y_common_%d_0p5deg.tif", year)))
  record("pm25_mmr_ug_per_kg_trailing5y", "historical_common", year, r)
}
rm(hist, hist05); gc()

scenario <- data.table(
  label = c("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
  ssp = c("ssp1", "ssp2", "ssp3", "ssp5"),
  exp = c("ssp126", "ssp245", "ssp370", "ssp585")
)
for (j in seq_len(nrow(scenario))) {
  r <- rast(file.path(native, sprintf("pm25_mmr_ugkg_trailing5y_%s_2025_2050.nc", scenario$exp[j])))
  crs(r) <- "EPSG:4326"
  r05 <- project(r, template, method = "bilinear")
  for (i in seq_along(2025:2050)) {
    year <- (2025:2050)[i]
    out <- r05[[i]]
    names(out) <- "pm25_mmr_ug_per_kg_trailing5y"
    write_new(
      out,
      file.path(future_dir, sprintf("pm25_mmr_ugkg_trailing5y_%s_%d_0p5deg.tif", scenario$ssp[j], year))
    )
    record("pm25_mmr_ug_per_kg_trailing5y", scenario$label[j], year, out)
  }
  rm(r, r05); gc()
}

fwrite(rbindlist(qa), file.path(qa_dir, "PM25_trailing5y_raster_QA_v2.csv"), bom = TRUE)
count_qa <- data.table(
  group = c("historical_2006_2024", "future_4ssp_2025_2050"),
  expected = c(19L, 104L),
  observed = c(
    length(list.files(hist_dir, pattern = "[.]tif$")),
    length(list.files(future_dir, pattern = "[.]tif$"))
  )
)
count_qa[, status := ifelse(expected == observed, "PASS", "FAIL")]
fwrite(count_qa, file.path(qa_dir, "PM25_trailing5y_file_count_QA_v2.csv"), bom = TRUE)
if (any(count_qa$status == "FAIL")) stop("PM2.5 v2 file-count QA failed")
cat("PASS\n")
