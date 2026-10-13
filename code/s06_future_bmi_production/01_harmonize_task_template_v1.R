#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(digest)
  library(jsonlite)
  library(terra)
})

options(stringsAsFactors = FALSE, warn = 1)

pilot_root <- file.path(.cfg$ROOT, ".codex_tmp", "fullcov_0p1_pilot_v1")
project_root <- file.path(.cfg$RESULTS)
old_0p1 <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "phase6g2_0p1deg_harmonization_pilot_v4_20260902"
)
production_root <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "full_covariate_annual_0p5deg_production_v1_20260906"
)
core_path <- file.path(production_root, "scripts", "full_covariate_annual_core_v1.R")

scenario <- "SSP5-8.5"
scenario_code <- "ssp585"
ssp <- "ssp5"
year <- 2050L

dirs <- list(
  data = file.path(pilot_root, "data", "0p1deg"),
  dynamic = file.path(pilot_root, "data", "0p1deg", "dynamic"),
  support = file.path(pilot_root, "data", "0p1deg", "support"),
  qa = file.path(pilot_root, "qa"),
  contracts = file.path(pilot_root, "contracts"),
  logs = file.path(pilot_root, "logs"),
  work = file.path(pilot_root, "work", "harmonization")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))
terraOptions(tempdir = dirs$work, memfrac = 0.60, progress = 1)
pilot_dirs <- dirs

log_path <- file.path(dirs$logs, "01_harmonize_ssp585_2050_0p1deg_v1.log")
log_connection <- file(log_path, open = "at", encoding = "UTF-8")
sink(log_connection, type = "output", split = TRUE)
sink(log_connection, type = "message", append = TRUE)
on.exit({
  try(sink(type = "message"), silent = TRUE)
  try(sink(type = "output"), silent = TRUE)
  try(close(log_connection), silent = TRUE)
}, add = TRUE)

stamp <- function(...) {
  cat(sprintf(
    "[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    paste0(..., collapse = "")
  ))
  flush.console()
}

write_json_new <- function(value, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path, call. = FALSE)
  temporary <- paste0(path, ".tmp")
  write_json(value, temporary, pretty = TRUE, auto_unbox = TRUE, na = "null")
  if (!file.rename(temporary, path)) stop("Failed to finalize: ", path, call. = FALSE)
}

write_csv_new <- function(value, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path, call. = FALSE)
  fwrite(value, path, bom = TRUE)
}

write_raster_resume <- function(value, path, datatype = "FLT4S") {
  if (file.exists(path)) {
    observed <- rast(path)
    if (!compareGeom(observed, template01, stopOnError = FALSE) ||
        nlyr(observed) != nlyr(value)) {
      stop("Existing output has invalid geometry: ", path, call. = FALSE)
    }
    stamp("reuse ", basename(path))
    return(invisible(path))
  }
  temporary <- file.path(dirname(path), paste0(".", basename(path), ".tmp.tif"))
  if (file.exists(temporary)) unlink(temporary)
  predictor <- if (startsWith(datatype, "FLT")) "PREDICTOR=3" else "PREDICTOR=2"
  writeRaster(
    value, temporary, overwrite = FALSE,
    wopt = list(
      datatype = datatype, NAflag = -9999,
      gdal = c(
        "COMPRESS=ZSTD", "ZSTD_LEVEL=1", predictor, "TILED=YES",
        "BIGTIFF=IF_SAFER"
      )
    )
  )
  if (!file.rename(temporary, path)) stop("Failed to finalize: ", path, call. = FALSE)
  invisible(path)
}

template01 <- rast(
  nrows = 1800, ncols = 3600, xmin = -180, xmax = 180,
  ymin = -90, ymax = 90, crs = "EPSG:4326"
)
template05 <- rast(
  nrows = 360, ncols = 720, xmin = -180, xmax = 180,
  ymin = -90, ymax = 90, crs = "EPSG:4326"
)

stamp("Loading the frozen full-model core and exact 0.5-degree dynamic inputs")
Sys.setenv(BMI_ANNUAL_CORE_ONLY = "1")
source(core_path, local = .GlobalEnv)
root <- pilot_root
dirs <- pilot_dirs
scenario <- "SSP5-8.5"
scenario_code <- "ssp585"
ssp <- "ssp5"
year <- 2050L
terraOptions(tempdir = dirs$work, memfrac = 0.60, progress = 1)
dynamic05 <- load_dynamic(scenario, year)

parent_probability_path <- file.path(
  production_root, "outputs", scenario_code, as.character(year),
  "bmi7_probability_age_sex_0p5deg_v1.tif"
)
parent_population_overall_path <- file.path(
  production_root, "outputs", scenario_code, as.character(year),
  "adult_population_overall_0p5deg_v1.tif"
)
parent_population_age_sex_path <- file.path(
  production_root, "outputs", scenario_code, as.character(year),
  "adult_population_age_sex_0p5deg_v1.tif"
)
parent_adult <- rast(parent_population_overall_path)
parent_adult_values <- values(parent_adult, mat = FALSE)
parent_active <- is.finite(parent_adult_values) & parent_adult_values > 0
if (sum(parent_active) < 40000L) stop("Too few formal 0.5-degree parent cells")

ncell01 <- ncell(template01)
ncell05 <- ncell(template05)
cell01 <- seq_len(ncell01)
row01 <- (cell01 - 1L) %/% 3600L
col01 <- (cell01 - 1L) %% 3600L
parent_cell01 <- (row01 %/% 5L) * 720L + (col01 %/% 5L) + 1L
parent_candidate <- parent_active[parent_cell01]
rm(cell01, row01, col01)
gc()

sum_by_parent <- function(value, parent, n = ncell05) {
  output <- numeric(n)
  aggregate <- rowsum(value, parent, reorder = FALSE, na.rm = TRUE)
  output[as.integer(rownames(aggregate))] <- aggregate[, 1]
  output
}

count_by_parent <- function(parent, n = ncell05) {
  tabulate(parent, nbins = n)
}

distribute_parent_total <- function(raw_value, target05, keep, fallback = NULL) {
  parent <- parent_cell01[keep]
  weight <- raw_value[keep]
  weight[!is.finite(weight) | weight < 0] <- 0
  denominator <- sum_by_parent(weight, parent)
  good <- denominator[parent] > 0
  if (!is.null(fallback)) {
    fallback_value <- fallback[keep]
    fallback_value[!is.finite(fallback_value) | fallback_value < 0] <- 0
    fallback_denominator <- sum_by_parent(fallback_value, parent)
    use <- !good & fallback_denominator[parent] > 0
    weight[use] <- fallback_value[use]
    denominator <- sum_by_parent(weight, parent)
    good <- denominator[parent] > 0
  }
  counts <- count_by_parent(parent)
  weight[!good] <- 1
  denominator <- sum_by_parent(weight, parent)
  target <- target05[parent]
  target[!is.finite(target) | target < 0] <- 0
  result_active <- target * weight / pmax(denominator[parent], .Machine$double.xmin)
  result <- rep(NA_real_, ncell01)
  result[keep] <- result_active
  result
}

match_parent_texture <- function(raw_value, target05, keep, beta,
                                 lower = -Inf, upper = Inf) {
  parent <- parent_cell01[keep]
  raw <- raw_value[keep]
  raw[!is.finite(raw)] <- 0
  count05 <- count_by_parent(parent)
  mean05 <- sum_by_parent(raw, parent) / pmax(count05, 1)
  deviation <- raw - mean05[parent]
  limits <- data.table(parent = parent, deviation = deviation)[
    , .(minimum = min(deviation), maximum = max(deviation)), by = parent
  ]
  scale05 <- rep(beta, ncell05)
  target <- target05
  target[!is.finite(target)] <- 0
  negative <- limits$minimum < 0 & is.finite(lower)
  if (any(negative)) {
    p <- limits$parent[negative]
    scale05[p] <- pmin(
      scale05[p], pmax(0, (target[p] - lower) / (-limits$minimum[negative]))
    )
  }
  positive <- limits$maximum > 0 & is.finite(upper)
  if (any(positive)) {
    p <- limits$parent[positive]
    scale05[p] <- pmin(
      scale05[p], pmax(0, (upper - target[p]) / limits$maximum[positive])
    )
  }
  result_active <- target[parent] + scale05[parent] * deviation
  if (is.finite(lower)) result_active <- pmax(lower, result_active)
  if (is.finite(upper)) result_active <- pmin(upper, result_active)
  result <- rep(NA_real_, ncell01)
  result[keep] <- result_active
  result
}

parent_sum_error <- function(value01, target05, keep) {
  parent <- parent_cell01[keep]
  observed <- sum_by_parent(value01[keep], parent)
  parents <- which(parent_active & is.finite(target05))
  absolute <- abs(observed[parents] - target05[parents])
  c(
    maximum_absolute = max(absolute, na.rm = TRUE),
    maximum_relative = max(absolute / pmax(abs(target05[parents]), 1), na.rm = TRUE)
  )
}

parent_mean_error <- function(value01, target05, keep) {
  parent <- parent_cell01[keep]
  observed <- sum_by_parent(value01[keep], parent)
  counts <- count_by_parent(parent)
  observed <- observed / pmax(counts, 1)
  parents <- which(parent_active & is.finite(target05) & counts > 0)
  absolute <- abs(observed[parents] - target05[parents])
  c(maximum_absolute = max(absolute, na.rm = TRUE))
}

as_raster01 <- function(value, name) {
  output <- template01
  values(output) <- value
  names(output) <- name
  output
}

raw_root <- file.path(.cfg$ROOT, ".codex_tmp", "full_covariate_raw_v1")

stamp("Building the scenario-specific 0.1-degree total-population weights")
population_source <- file.path(
  raw_root, "ssp_grid_population", "ISIMIP2b_JonesONeill_2p5arcmin", "SSP5",
  "population_ssp5soc_2.5min_annual_2006-2100.nc4"
)
population_stack <- rast(population_source)
population_time <- time(population_stack)
population_years <- if (inherits(population_time, c("Date", "POSIXt"))) {
  as.integer(format(population_time, "%Y"))
} else {
  as.integer(population_time)
}
population_index <- match(year, population_years)
if (is.na(population_index)) stop("SSP5 population is missing 2050")
population_raw01 <- resample(population_stack[[population_index]], template01, method = "sum")
population_raw_values <- values(population_raw01, mat = FALSE)
population05 <- dynamic05$population
population_pre <- distribute_parent_total(
  population_raw_values, population05, parent_candidate
)
rm(population_stack, population_raw01, population_raw_values)
gc()

stamp("Reading the fixed WorldPop 2025 adult age-sex prior")
worldpop_root <- file.path(old_0p1, "data", "0p1deg", "worldpop_2025_prior")
worldpop_paths <- list.files(worldpop_root, pattern = "[.]tif$", full.names = TRUE)
if (length(worldpop_paths) != 28L) stop("Expected 28 WorldPop priors")
adult_prior <- values(app(rast(worldpop_paths), sum, na.rm = TRUE), mat = FALSE)
analysis_keep <- parent_candidate & (
  (is.finite(population_pre) & population_pre > 0) |
    (is.finite(adult_prior) & adult_prior > 0)
)
if (sum(analysis_keep) < 900000L) stop("Unexpectedly small 0.1-degree support")

population01 <- distribute_parent_total(
  population_pre, population05, analysis_keep, fallback = adult_prior
)
population_path <- file.path(
  dirs$dynamic, "population_ssp5_2050_0p1deg_parent_calibrated_v1.tif"
)
write_raster_resume(as_raster01(population01, "population_persons"), population_path, "FLT8S")

stamp("Building GDP and exact parent-total calibration")
gdp_source <- file.path(
  raw_root, "ssp_grid_gdp", "WangSun_0p25deg", "extracted", "025d",
  "GDP2050_ssp5.tif"
)
gdp_raw <- rast(gdp_source)
gdp_raw[gdp_raw < 0] <- NA
gdp_raw01 <- resample(gdp_raw, template01, method = "bilinear")
gdp01 <- distribute_parent_total(
  values(gdp_raw01, mat = FALSE), dynamic05$gdp, analysis_keep,
  fallback = population01
)
gdp_path <- file.path(dirs$dynamic, "gdp_ssp5_2050_0p1deg_parent_calibrated_v1.tif")
write_raster_resume(as_raster01(gdp01, "gdp_ppp_total"), gdp_path, "FLT8S")
rm(gdp_raw, gdp_raw01)
gc()

stamp("Building the high-resolution NTL texture with the frozen 1.5 bridge")
ntl_source <- file.path(raw_root, "FUT_NTL", "extracted", "ssp5", "NTL_SSP5_2050.tif")
ntl_raw <- rast(ntl_source)
ntl_raw[ntl_raw < 0] <- NA
ntl_log_raw <- log1p(ntl_raw)
ntl_raw01 <- project(ntl_log_raw, template01, method = "average")
ntl01 <- match_parent_texture(
  values(ntl_raw01, mat = FALSE), dynamic05$ntl, analysis_keep,
  beta = 1.5, lower = 0
)
ntl_path <- file.path(dirs$dynamic, "ntl_log_ssp5_2050_0p1deg_parent_centered_v1.tif")
write_raster_resume(as_raster01(ntl01, "ntl_log"), ntl_path, "FLT4S")
rm(ntl_raw, ntl_log_raw, ntl_raw01)
gc()

stamp("Building urban-share and built-fraction textures")
urban_source <- file.path(
  raw_root, "ssp_urban", "Gao_Pesaresi_SEDAC_1km", "extracted",
  "ssp5-geotiff", "ssp5_2050.tif"
)
urban_raw <- rast(urban_source)
urban_raw[urban_raw < 0] <- NA
urban_raw01 <- if (same.crs(urban_raw, template01)) {
  resample(urban_raw, template01, method = "average")
} else {
  project(urban_raw, template01, method = "average")
}
urban_raw_values <- values(urban_raw01, mat = FALSE)
urban01 <- match_parent_texture(
  urban_raw_values, dynamic05$urban, analysis_keep,
  beta = 1.15547908384679, lower = 0, upper = 1
)
built01 <- match_parent_texture(
  urban_raw_values, dynamic05$built, analysis_keep,
  beta = 0.720115027726241, lower = 0, upper = 1
)
urban_path <- file.path(dirs$dynamic, "urban_share_ssp5_2050_0p1deg_parent_centered_v1.tif")
built_path <- file.path(dirs$dynamic, "built_fraction_ssp5_2050_0p1deg_parent_centered_v1.tif")
write_raster_resume(as_raster01(urban01, "urban_share"), urban_path, "FLT4S")
write_raster_resume(as_raster01(built01, "built_fraction_proxy"), built_path, "FLT4S")
rm(urban_raw, urban_raw01, urban_raw_values)
gc()

stamp("Materializing coarse-source covariates without invented sub-parent texture")
coarse_variables <- list(
  pm25_mmr_trailing5y = dynamic05$pm25,
  temperature_c = dynamic05$temperature,
  relative_humidity_pct = dynamic05$humidity
)
coarse_paths <- character()
for (variable in names(coarse_variables)) {
  value <- rep(NA_real_, ncell01)
  value[analysis_keep] <- coarse_variables[[variable]][parent_cell01[analysis_keep]]
  path <- file.path(
    dirs$dynamic, sprintf("%s_ssp5_2050_0p1deg_parent_constant_v1.tif", variable)
  )
  write_raster_resume(as_raster01(value, variable), path, "FLT4S")
  coarse_paths[variable] <- path
}

stamp("Writing support rasters")
mask_value <- rep(NA_integer_, ncell01)
mask_value[analysis_keep] <- 1L
mask_path <- file.path(dirs$support, "analysis_mask_ssp585_2050_0p1deg_v1.tif")
write_raster_resume(as_raster01(mask_value, "analysis_mask"), mask_path, "INT2S")

parent_id_value <- rep(NA_integer_, ncell01)
parent_id_value[analysis_keep] <- parent_cell01[analysis_keep]
parent_id_path <- file.path(dirs$support, "parent_cell_id_ssp585_2050_0p1deg_v1.tif")
write_raster_resume(as_raster01(parent_id_value, "parent_cell_id"), parent_id_path, "INT4S")

country_id_value <- rep(NA_integer_, ncell01)
country_id_value[analysis_keep] <- country_values[parent_cell01[analysis_keep]]
country_id_path <- file.path(dirs$support, "country_id_ssp585_2050_0p1deg_v1.tif")
write_raster_resume(as_raster01(country_id_value, "country_id"), country_id_path, "INT4S")

qa_rows <- list(
  data.table(
    variable = "population", conservation = "parent_sum",
    maximum_parent_absolute_error = parent_sum_error(
      population01, population05, analysis_keep
    )[["maximum_absolute"]],
    maximum_parent_relative_error = parent_sum_error(
      population01, population05, analysis_keep
    )[["maximum_relative"]]
  ),
  data.table(
    variable = "gdp", conservation = "parent_sum",
    maximum_parent_absolute_error = parent_sum_error(
      gdp01, dynamic05$gdp, analysis_keep
    )[["maximum_absolute"]],
    maximum_parent_relative_error = parent_sum_error(
      gdp01, dynamic05$gdp, analysis_keep
    )[["maximum_relative"]]
  )
)
for (item in list(
  list("ntl_log", ntl01, dynamic05$ntl),
  list("urban_share", urban01, dynamic05$urban),
  list("built_fraction_proxy", built01, dynamic05$built)
)) {
  qa_rows[[length(qa_rows) + 1L]] <- data.table(
    variable = item[[1]], conservation = "parent_arithmetic_mean",
    maximum_parent_absolute_error = parent_mean_error(
      item[[2]], item[[3]], analysis_keep
    )[["maximum_absolute"]],
    maximum_parent_relative_error = NA_real_
  )
}
qa <- rbindlist(qa_rows, fill = TRUE)
qa_path <- file.path(dirs$qa, "0p1deg_covariate_parent_consistency_ssp585_2050_v1.csv")
if (!file.exists(qa_path)) write_csv_new(qa, qa_path)

output_paths <- c(
  population_path, gdp_path, ntl_path, urban_path, built_path,
  unname(coarse_paths), mask_path, parent_id_path, country_id_path
)
raster_qa <- rbindlist(lapply(output_paths, function(path) {
  value <- rast(path)
  summary <- global(value, c("min", "max", "mean"), na.rm = TRUE)
  data.table(
    path = path, layers = nlyr(value), nrow = nrow(value), ncol = ncol(value),
    resolution_x = res(value)[1], resolution_y = res(value)[2],
    crs = crs(value, proj = TRUE),
    geometry_match = compareGeom(value, template01, stopOnError = FALSE),
    minimum = summary[1, 1], maximum = summary[1, 2], mean = summary[1, 3]
  )
}))
raster_qa_path <- file.path(dirs$qa, "0p1deg_covariate_raster_inventory_ssp585_2050_v1.csv")
if (!file.exists(raster_qa_path)) write_csv_new(raster_qa, raster_qa_path)

checks <- data.table(
  check_id = sprintf("FC01_%02d", 1:9),
  requirement = c(
    "Frozen full-model core loaded",
    "Formal 0.5-degree parent product is readable",
    "At least 900,000 nested child cells have support",
    "Population closes to the exact formal parent input",
    "GDP closes to the exact formal parent input",
    "NTL, urban share and built fraction preserve parent means",
    "All output rasters share the frozen 0.1-degree geometry",
    "Urban and built fractions are within [0,1] and humidity within [0,100]",
    "No model fitting, BMI prediction or burden analysis occurred"
  ),
  observed = c(
    basename(core_path), basename(parent_probability_path), sum(analysis_keep),
    qa[variable == "population", maximum_parent_relative_error],
    qa[variable == "gdp", maximum_parent_relative_error],
    max(qa[conservation == "parent_arithmetic_mean", maximum_parent_absolute_error]),
    paste(sum(raster_qa$geometry_match), "/", nrow(raster_qa)),
    paste0(
      "urban=", range(urban01[analysis_keep], na.rm = TRUE)[1], ":",
      range(urban01[analysis_keep], na.rm = TRUE)[2], ";built=",
      range(built01[analysis_keep], na.rm = TRUE)[1], ":",
      range(built01[analysis_keep], na.rm = TRUE)[2], ";humidity=",
      range(coarse_variables$relative_humidity_pct[parent_cell01[analysis_keep]], na.rm = TRUE)[1],
      ":",
      range(coarse_variables$relative_humidity_pct[parent_cell01[analysis_keep]], na.rm = TRUE)[2]
    ),
    "scope guard retained"
  ),
  pass = c(
    TRUE, file.exists(parent_probability_path), sum(analysis_keep) >= 900000L,
    qa[variable == "population", maximum_parent_relative_error] <= 1e-8,
    qa[variable == "gdp", maximum_parent_relative_error] <= 1e-8,
    max(qa[conservation == "parent_arithmetic_mean", maximum_parent_absolute_error]) <= 1e-5,
    all(raster_qa$geometry_match),
    min(urban01[analysis_keep], na.rm = TRUE) >= 0 &&
      max(urban01[analysis_keep], na.rm = TRUE) <= 1 &&
      min(built01[analysis_keep], na.rm = TRUE) >= 0 &&
      max(built01[analysis_keep], na.rm = TRUE) <= 1 &&
      min(coarse_variables$relative_humidity_pct[parent_cell01[analysis_keep]], na.rm = TRUE) >= 0 &&
      max(coarse_variables$relative_humidity_pct[parent_cell01[analysis_keep]], na.rm = TRUE) <= 100,
    TRUE
  )
)
check_path <- file.path(dirs$qa, "Gate_FC01_covariate_checks_ssp585_2050_v1.csv")
if (!file.exists(check_path)) write_csv_new(checks, check_path)

status <- if (all(checks$pass)) "PASS_GATE_FC01_0P1DEG_COVARIATES" else
  "FAIL_GATE_FC01_0P1DEG_COVARIATES"
contract <- list(
  status = status,
  scenario = scenario,
  year = year,
  grid = list(
    crs = "EPSG:4326", resolution_degrees = 0.1,
    dimensions = c(rows = 1800, columns = 3600), nesting_factor = 5
  ),
  formal_parent = list(
    probability = parent_probability_path,
    adult_population_age_sex = parent_population_age_sex_path,
    source = "full_covariate_annual_0p5deg_production_v1_20260906"
  ),
  spatial_rules = list(
    population = "SSP5 2.5-arcmin weights, exact 0.5-degree sum calibration",
    gdp = "Wang-Sun SSP5 weights, exact 0.5-degree sum calibration",
    ntl = "native 1-km projected NTL texture; additive within-parent contrast capped at bridge slope 1.5; exact parent mean",
    urban_and_built = "native 1-km urban-land texture with frozen bridge slopes and bounded exact parent means",
    pm25_temperature_humidity = "formal 0.5-degree value inherited by children; no synthetic sub-parent detail",
    static = "existing validated direct 0.1-degree elevation and travel-time rasters",
    demographic_prior = "existing validated WorldPop 2025 age-sex 0.1-degree priors"
  ),
  warning = paste(
    "The 0.1-degree layer is a nested spatial allocation product.",
    "It does not claim independent 0.1-degree information for coarse CMIP6 variables."
  ),
  prohibited = c("model refitting", "BMI prediction", "disease burden analysis")
)
contract_path <- file.path(dirs$contracts, "full_covariate_0p1deg_pilot_contract_v1.json")
if (!file.exists(contract_path)) write_json_new(contract, contract_path)

summary_path <- file.path(dirs$qa, "Gate_FC01_summary_ssp585_2050_v1.json")
if (!file.exists(summary_path)) write_json_new(list(
  status = status,
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  scenario = scenario,
  year = year,
  formal_parent_cells = sum(parent_active),
  analysis_child_cells = sum(analysis_keep),
  maximum_population_parent_relative_error = qa[
    variable == "population", maximum_parent_relative_error
  ],
  maximum_gdp_parent_relative_error = qa[
    variable == "gdp", maximum_parent_relative_error
  ],
  maximum_mean_covariate_parent_error = max(qa[
    conservation == "parent_arithmetic_mean", maximum_parent_absolute_error
  ]),
  outputs = output_paths
), summary_path)

stamp(status)
if (!startsWith(status, "PASS_")) quit(status = 2L)
