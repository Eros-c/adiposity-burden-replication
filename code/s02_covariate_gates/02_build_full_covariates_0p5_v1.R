#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(jsonlite)
})

options(stringsAsFactors = FALSE, warn = 1)
terraOptions(progress = 1, memfrac = 0.65)

run_dir <- Sys.getenv("FULLCOV_RUN_DIR")
if (!nzchar(run_dir)) stop("FULLCOV_RUN_DIR is required")

raw_root <- file.path(.cfg$ROOT, ".codex_tmp", "full_covariate_raw_v1")
historic_root <- file.path(.cfg$ROOT, ".codex_tmp", "historic_results_v1")
future_root <- file.path(.cfg$ROOT, ".codex_tmp", "future_results_v1")
gate_c <- paste0(
  future_root,
  "/02_harmonized_covariates/gate_c_v1_country_coverage_remediated_20260828_122619"
)

scenarios <- data.table(
  scenario = c("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
  ssp = c("ssp1", "ssp2", "ssp3", "ssp5"),
  experiment = c("ssp126", "ssp245", "ssp370", "ssp585")
)
years_future <- 2025:2050

dirs <- list(
  data = file.path(run_dir, "data", "0p5deg"),
  pm_hist = file.path(run_dir, "data", "0p5deg", "pm25_mmr", "historical_common"),
  pm_future = file.path(run_dir, "data", "0p5deg", "pm25_mmr", "future"),
  ntl = file.path(run_dir, "data", "0p5deg", "ntl_log"),
  urban = file.path(run_dir, "data", "0p5deg", "urban_share"),
  built = file.path(run_dir, "data", "0p5deg", "built_fraction_proxy"),
  work = file.path(run_dir, "work"),
  qa = file.path(run_dir, "qa"),
  contracts = file.path(run_dir, "contracts"),
  logs = file.path(run_dir, "logs")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

log_path <- file.path(dirs$logs, "full_covariate_harmonization_0p5_v1.log")
log_con <- file(log_path, open = "wt", encoding = "UTF-8")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message", append = TRUE)
on.exit({
  try(sink(type = "message"), silent = TRUE)
  try(sink(type = "output"), silent = TRUE)
  try(close(log_con), silent = TRUE)
}, add = TRUE)

stamp <- function(...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), paste0(..., collapse = "")))
  flush.console()
}

write_new <- function(r, path, datatype = "FLT4S") {
  if (file.exists(path)) stop("Refusing to overwrite: ", path)
  writeRaster(
    r, path, overwrite = FALSE,
    wopt = list(
      datatype = datatype, NAflag = -9999,
      gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES", "BIGTIFF=IF_SAFER")
    )
  )
  path
}

logit_r <- function(r, eps = 1e-6) {
  x <- clamp(r, eps, 1 - eps)
  log(x / (1 - x))
}

invlogit_r <- function(x) 1 / (1 + exp(-x))

interp_r <- function(r0, r1, w) (1 - w) * r0 + w * r1

template <- rast(file.path(gate_c, "data/0p5deg/country_id_0p5deg.tif"))
template[] <- NA
crs(template) <- "EPSG:4326"

hist_2020_path <- file.path(
  historic_root,
  "cache/auxiliary_mrp_0p5deg/covariates/global_covariates_2020_0p5deg.tif"
)
hist_2024_path <- file.path(
  historic_root,
  "cache/auxiliary_mrp_0p5deg/covariates/global_covariates_2024_0p5deg.tif"
)
if (!file.exists(hist_2020_path) || !file.exists(hist_2024_path)) {
  stop("Historical 0.5-degree covariate stacks are missing")
}
hist20 <- rast(hist_2020_path)
hist24 <- rast(hist_2024_path)
hist_ntl20 <- resample(hist20[["ntl_log"]], template, method = "bilinear")
hist_ntl24 <- resample(hist24[["ntl_log"]], template, method = "bilinear")
hist_urban20 <- resample(hist20[["urban_share"]], template, method = "bilinear")
hist_built20 <- resample(hist20[["built_fraction"]], template, method = "bilinear")
pop25 <- rast(file.path(gate_c, "data/0p5deg/population/population_ssp2_2025_0p5deg.tif"))

qa_rows <- list()
qa_i <- 0L
add_qa <- function(variable, scenario, year, r, role) {
  g <- global(r, c("min", "max", "mean"), na.rm = TRUE)
  qa_i <<- qa_i + 1L
  qa_rows[[qa_i]] <<- data.table(
    variable = variable, scenario = scenario, year = year, role = role,
    min = g[1, "min"], max = g[1, "max"], mean = g[1, "mean"],
    finite_cells = global(ifel(is.finite(r), 1, NA), "sum", na.rm = TRUE)[1, 1]
  )
}

stamp("Building PM2.5 near-surface mass-mixing-ratio rasters")
pm_native <- file.path(run_dir, "data", "pm25_annual_native")
pm_hist_nc <- file.path(pm_native, "pm25_mmr_ugkg_historical_common_1990_2024.nc")
pm_hist <- rast(pm_hist_nc)
crs(pm_hist) <- "EPSG:4326"
pm_hist05 <- project(pm_hist, template, method = "bilinear")
for (year in 2006:2024) {
  idx <- year - 1990 + 1L
  out <- pm_hist05[[idx]]
  names(out) <- "pm25_mmr_ug_per_kg"
  path <- file.path(dirs$pm_hist, sprintf("pm25_mmr_ugkg_common_%d_0p5deg.tif", year))
  write_new(out, path)
  add_qa("pm25_mmr_ug_per_kg", "historical_common", year, out, "model_refit")
}
rm(pm_hist, pm_hist05); gc()

for (j in seq_len(nrow(scenarios))) {
  exp <- scenarios$experiment[j]
  nc <- file.path(pm_native, sprintf("pm25_mmr_ugkg_%s_2025_2050.nc", exp))
  r <- rast(nc)
  crs(r) <- "EPSG:4326"
  r05 <- project(r, template, method = "bilinear")
  for (i in seq_along(years_future)) {
    year <- years_future[i]
    out <- r05[[i]]
    names(out) <- "pm25_mmr_ug_per_kg"
    path <- file.path(
      dirs$pm_future,
      sprintf("pm25_mmr_ugkg_%s_%d_0p5deg.tif", scenarios$ssp[j], year)
    )
    write_new(out, path)
    add_qa("pm25_mmr_ug_per_kg", scenarios$scenario[j], year, out, "future")
  }
  rm(r, r05); gc()
}

stamp("Building raw future NTL nodes at 0.5 degrees")
ntl_nodes <- c(2020, 2025, 2030, 2035, 2040, 2045, 2050)
ntl_node_paths <- list()
for (j in seq_len(nrow(scenarios))) {
  for (year in ntl_nodes) {
    src <- file.path(
      raw_root, "FUT_NTL/extracted", scenarios$ssp[j],
      sprintf("NTL_%s_%d.tif", toupper(scenarios$ssp[j]), year)
    )
    if (!file.exists(src)) stop("Missing future NTL node: ", src)
    stamp("  NTL node ", scenarios$scenario[j], " ", year)
    r <- rast(src)
    r[r < 0] <- NA
    r <- log1p(r)
    out <- project(r, template, method = "average")
    names(out) <- "ntl_log_raw"
    path <- file.path(dirs$work, sprintf("ntl_log_raw_%s_%d_0p5deg.tif", scenarios$ssp[j], year))
    write_new(out, path)
    ntl_node_paths[[paste(scenarios$ssp[j], year, sep = "_")]] <- path
    rm(r, out); gc()
  }
}

# Base R has no weighted cor method; compute explicitly.
weighted_cor <- function(x, y, w) {
  mx <- weighted.mean(x, w); my <- weighted.mean(y, w)
  sum(w * (x - mx) * (y - my)) /
    sqrt(sum(w * (x - mx)^2) * sum(w * (y - my)^2))
}

weighted_bridge <- function(observed, projected, weight, name) {
  s <- c(observed, projected, weight)
  names(s) <- c("observed", "projected", "weight")
  d <- as.data.table(values(s, dataframe = TRUE, na.rm = TRUE))
  d <- d[is.finite(observed) & is.finite(projected) & is.finite(weight) & weight > 0]
  if (nrow(d) < 100) stop("Insufficient overlap for bridge: ", name)
  cap <- quantile(d$weight, 0.99, na.rm = TRUE)
  d[, w := sqrt(pmin(weight, cap))]
  fit <- lm(observed ~ projected, data = d, weights = w)
  pred <- predict(fit, newdata = d)
  list(
    summary = data.table(
      bridge = name, n = nrow(d), intercept = unname(coef(fit)[1]),
      slope = unname(coef(fit)[2]), weighted_correlation = weighted_cor(d$observed, d$projected, d$w),
      rmse = sqrt(weighted.mean((d$observed - pred)^2, d$w))
    ),
    slope = unname(coef(fit)[2])
  )
}

ntl_raw20 <- rast(ntl_node_paths[["ssp2_2020"]])
ntl_bridge <- weighted_bridge(hist_ntl20, ntl_raw20, pop25, "future_NTL_to_historical_ntl_log")
ntl_beta <- max(0, min(2, ntl_bridge$slope))

stamp("Interpolating and anchoring annual future NTL")
for (j in seq_len(nrow(scenarios))) {
  raw20 <- rast(ntl_node_paths[[paste(scenarios$ssp[j], 2020, sep = "_")]])
  raw25 <- rast(ntl_node_paths[[paste(scenarios$ssp[j], 2025, sep = "_")]])
  raw24 <- interp_r(raw20, raw25, 4 / 5)
  for (year in years_future) {
    y0 <- max(ntl_nodes[ntl_nodes <= year])
    y1 <- min(ntl_nodes[ntl_nodes >= year])
    r0 <- rast(ntl_node_paths[[paste(scenarios$ssp[j], y0, sep = "_")]])
    raw_y <- if (y1 == y0) r0 else {
      r1 <- rast(ntl_node_paths[[paste(scenarios$ssp[j], y1, sep = "_")]])
      interp_r(r0, r1, (year - y0) / (y1 - y0))
    }
    out <- hist_ntl24 + ntl_beta * (raw_y - raw24)
    out <- clamp(out, 0, Inf)
    names(out) <- "ntl_log"
    path <- file.path(dirs$ntl, sprintf("ntl_log_%s_%d_0p5deg.tif", scenarios$ssp[j], year))
    write_new(out, path)
    add_qa("ntl_log", scenarios$scenario[j], year, out, "future_2024_anchored")
    rm(r0, raw_y, out); gc()
  }
  rm(raw20, raw25, raw24); gc()
}

stamp("Building urban-share and built-fraction future proxies")
urban_node_years <- c(2020, 2030, 2040, 2050)
urban_node_paths <- list()
for (j in seq_len(nrow(scenarios))) {
  for (year in urban_node_years) {
    src <- file.path(
      raw_root, "ssp_urban/Gao_Pesaresi_SEDAC_1km/extracted",
      sprintf("%s-geotiff/%s_%d.tif", scenarios$ssp[j], scenarios$ssp[j], year)
    )
    if (!file.exists(src)) stop("Missing SSP urban-land node: ", src)
    r <- rast(src); r[r < 0] <- NA
    out <- project(r, template, method = "average")
    out <- clamp(out, 1e-6, 1 - 1e-6)
    names(out) <- "urban_land_fraction_raw"
    path <- file.path(dirs$work, sprintf("urban_land_raw_%s_%d_0p5deg.tif", scenarios$ssp[j], year))
    write_new(out, path)
    urban_node_paths[[paste(scenarios$ssp[j], year, sep = "_")]] <- path
    rm(r, out); gc()
  }
}

urban_raw20 <- rast(urban_node_paths[["ssp2_2020"]])
urban_bridge <- weighted_bridge(logit_r(hist_urban20), logit_r(urban_raw20), pop25,
                                "SSP_urban_land_to_historical_urban_share")
built_bridge <- weighted_bridge(logit_r(hist_built20), logit_r(urban_raw20), pop25,
                                "SSP_urban_land_to_historical_built_fraction")
urban_beta <- max(0, min(2, urban_bridge$slope))
built_beta <- max(0, min(2, built_bridge$slope))

for (j in seq_len(nrow(scenarios))) {
  raw20 <- logit_r(rast(urban_node_paths[[paste(scenarios$ssp[j], 2020, sep = "_")]]))
  raw30 <- logit_r(rast(urban_node_paths[[paste(scenarios$ssp[j], 2030, sep = "_")]]))
  raw24 <- interp_r(raw20, raw30, 4 / 10)
  for (year in years_future) {
    y0 <- max(urban_node_years[urban_node_years <= year])
    y1 <- min(urban_node_years[urban_node_years >= year])
    r0 <- logit_r(rast(urban_node_paths[[paste(scenarios$ssp[j], y0, sep = "_")]]))
    raw_y <- if (y1 == y0) r0 else {
      r1 <- logit_r(rast(urban_node_paths[[paste(scenarios$ssp[j], y1, sep = "_")]]))
      interp_r(r0, r1, (year - y0) / (y1 - y0))
    }
    urban_out <- invlogit_r(logit_r(hist_urban20) + urban_beta * (raw_y - raw24))
    built_out <- invlogit_r(logit_r(hist_built20) + built_beta * (raw_y - raw24))
    names(urban_out) <- "urban_share"
    names(built_out) <- "built_fraction_proxy"
    up <- file.path(dirs$urban, sprintf("urban_share_%s_%d_0p5deg.tif", scenarios$ssp[j], year))
    bp <- file.path(dirs$built, sprintf("built_fraction_proxy_%s_%d_0p5deg.tif", scenarios$ssp[j], year))
    write_new(urban_out, up)
    write_new(built_out, bp)
    add_qa("urban_share", scenarios$scenario[j], year, urban_out, "SSP_urban_land_change_proxy")
    add_qa("built_fraction_proxy", scenarios$scenario[j], year, built_out,
           "SSP_urban_land_change_proxy")
    rm(r0, raw_y, urban_out, built_out); gc()
  }
  rm(raw20, raw30, raw24); gc()
}

bridge_qa <- rbindlist(list(ntl_bridge$summary, urban_bridge$summary, built_bridge$summary), fill = TRUE)
bridge_qa[, slope_used := c(ntl_beta, urban_beta, built_beta)]
fwrite(bridge_qa, file.path(dirs$qa, "Full_covariate_bridge_QA_v1.csv"), bom = TRUE)

raster_qa <- rbindlist(qa_rows, fill = TRUE)
fwrite(raster_qa, file.path(dirs$qa, "Full_covariate_raster_QA_v1.csv"), bom = TRUE)

expected <- data.table(
  variable = c("pm25_historical", "pm25_future", "ntl_future", "urban_share_future", "built_proxy_future"),
  expected = c(19L, 104L, 104L, 104L, 104L),
  observed = c(
    length(list.files(dirs$pm_hist, pattern = "[.]tif$")),
    length(list.files(dirs$pm_future, pattern = "[.]tif$")),
    length(list.files(dirs$ntl, pattern = "[.]tif$")),
    length(list.files(dirs$urban, pattern = "[.]tif$")),
    length(list.files(dirs$built, pattern = "[.]tif$"))
  )
)
expected[, status := ifelse(expected == observed, "PASS", "FAIL")]
fwrite(expected, file.path(dirs$qa, "Full_covariate_expected_file_count_QA_v1.csv"), bom = TRUE)
if (any(expected$status == "FAIL")) stop("Output file-count QA failed")

contract <- list(
  status = "PASS_WITH_DOCUMENTED_PROXIES",
  primary_grid = list(crs = "EPSG:4326", resolution = 0.5, extent = c(-180, 180, -90, 90)),
  pm25 = list(
    variable = "near-surface dry-aerosol PM2.5 mass mixing ratio",
    unit = "microgram per kilogram of air",
    rule = "new coefficient must be refitted; never reuse concentration coefficient",
    source_model = "NorESM2-LM r1i1p1f1"
  ),
  ntl = list(
    transform = "log1p before area averaging",
    temporal_interpolation = "linear between official five-year nodes on log1p scale",
    bridge = "scenario changes anchored to observed 2024 NTL with overlap-derived slope"
  ),
  urban_share = list(
    source = "SSP urban land extent fraction",
    interpretation = "proxy for historical GHS-SMOD urban area share",
    bridge = "logit-scale change anchored to historical 2020/2024 level"
  ),
  built_fraction = list(
    source = "SSP urban land extent fraction",
    interpretation = "proxy trajectory, not a direct future built-surface observation",
    bridge = "logit-scale change anchored to historical GHS built-surface fraction",
    sensitivity_requirement = "compare models with and without this proxy because urban_share and built proxy share a source"
  ),
  no_model_fitting_authorized_by_this_script = TRUE
)
write_json(contract, file.path(dirs$contracts, "Full_covariate_harmonization_contract_v1.json"),
           pretty = TRUE, auto_unbox = TRUE)

unlink(dirs$work, recursive = TRUE, force = TRUE)
stamp("PASS: full covariate 0.5-degree harmonization completed")
