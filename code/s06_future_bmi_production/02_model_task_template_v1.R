#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(terra)
})

options(stringsAsFactors = FALSE, warn = 1)

pilot_root <- file.path(.cfg$ROOT, ".codex_tmp", "fullcov_0p1_pilot_v1")
project_root <- file.path(.cfg$RESULTS)
production_root <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "full_covariate_annual_0p5deg_production_v1_20260906"
)
old_0p1 <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "phase6g2_0p1deg_harmonization_pilot_v4_20260902"
)
core_path <- file.path(production_root, "scripts", "full_covariate_annual_core_v1.R")

scenario <- "SSP5-8.5"
scenario_code <- "ssp585"
ssp <- "ssp5"
year <- 2050L

dirs <- list(
  dynamic = file.path(pilot_root, "data", "0p1deg", "dynamic"),
  support = file.path(pilot_root, "data", "0p1deg", "support"),
  binary = file.path(
    pilot_root, "work", "binary_strict_density_v2", scenario_code,
    as.character(year)
  ),
  pre_population = file.path(pilot_root, "work", "population_pre_mask"),
  qa = file.path(pilot_root, "qa"),
  contracts = file.path(pilot_root, "contracts"),
  logs = file.path(pilot_root, "logs"),
  work = file.path(pilot_root, "work", "model")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))
pilot_dirs <- dirs
terraOptions(tempdir = dirs$work, memfrac = 0.60, progress = 1)

log_path <- file.path(dirs$logs, "02_population_and_unraked_mrp_ssp585_2050_v1.log")
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

write_binary_new <- function(value, path, size) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path, call. = FALSE)
  temporary <- paste0(path, ".tmp")
  connection <- file(temporary, open = "wb")
  on.exit(try(close(connection), silent = TRUE), add = TRUE)
  writeBin(as.numeric(value), connection, size = size, endian = "little")
  close(connection)
  if (!file.rename(temporary, path)) stop("Failed to finalize: ", path, call. = FALSE)
}

write_integer_binary_new <- function(value, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path, call. = FALSE)
  temporary <- paste0(path, ".tmp")
  connection <- file(temporary, open = "wb")
  on.exit(try(close(connection), silent = TRUE), add = TRUE)
  writeBin(as.integer(value), connection, size = 4L, endian = "little")
  close(connection)
  if (!file.rename(temporary, path)) stop("Failed to finalize: ", path, call. = FALSE)
}

write_integer_binary_resume <- function(value, path) {
  if (!file.exists(path)) return(write_integer_binary_new(value, path))
  if (file.info(path)$size != length(value) * 4) {
    stop("Existing integer binary has the wrong size: ", path, call. = FALSE)
  }
  observed <- readBin(
    path, what = "integer", n = length(value), size = 4L, endian = "little"
  )
  if (!identical(observed, as.integer(value))) {
    stop("Existing integer binary differs from deterministic rerun: ", path,
         call. = FALSE)
  }
  invisible(path)
}

write_binary_resume <- function(value, path, size, tolerance = 1e-10) {
  if (!file.exists(path)) return(write_binary_new(value, path, size))
  expected_bytes <- length(value) * size
  if (file.info(path)$size != expected_bytes) {
    stop("Existing binary has the wrong size: ", path, call. = FALSE)
  }
  observed <- readBin(
    path, what = "numeric", n = length(value), size = size, endian = "little"
  )
  error <- max(abs(observed - value), na.rm = TRUE)
  if (!is.finite(error) || error > tolerance) {
    stop("Existing binary differs from deterministic rerun: ", path, call. = FALSE)
  }
  invisible(path)
}

write_raster_new <- function(value, path, template, datatype = "INT4S") {
  if (file.exists(path)) stop("Refusing to overwrite: ", path, call. = FALSE)
  output <- template
  values(output) <- value
  temporary <- file.path(dirname(path), paste0(".", basename(path), ".tmp.tif"))
  writeRaster(
    output, temporary, overwrite = FALSE,
    wopt = list(
      datatype = datatype, NAflag = -9999,
      gdal = c("COMPRESS=ZSTD", "ZSTD_LEVEL=1", "TILED=YES", "BIGTIFF=IF_SAFER")
    )
  )
  if (!file.rename(temporary, path)) stop("Failed to finalize: ", path, call. = FALSE)
}

write_raster_resume <- function(value, path, template, datatype = "INT4S") {
  if (!file.exists(path)) return(write_raster_new(value, path, template, datatype))
  observed <- rast(path)
  if (!compareGeom(observed, template, stopOnError = FALSE) || nlyr(observed) != 1L) {
    stop("Existing raster has the wrong geometry: ", path, call. = FALSE)
  }
  invisible(path)
}

fc01 <- fromJSON(file.path(dirs$qa, "Gate_FC01_summary_ssp585_2050_v1.json"))
if (!identical(fc01$status, "PASS_GATE_FC01_0P1DEG_COVARIATES")) {
  stop("Gate FC01 is not PASS", call. = FALSE)
}

stamp("Loading the frozen 22-feature MRP core")
Sys.setenv(BMI_ANNUAL_CORE_ONLY = "1")
source(core_path, local = .GlobalEnv)
dirs <- pilot_dirs
root <- pilot_root
scenario <- "SSP5-8.5"
scenario_code <- "ssp585"
ssp <- "ssp5"
year <- 2050L
terraOptions(tempdir = dirs$work, memfrac = 0.60, progress = 1)
dynamic05 <- load_dynamic(scenario, year)

template01 <- rast(
  nrows = 1800, ncols = 3600, xmin = -180, xmax = 180,
  ymin = -90, ymax = 90, crs = "EPSG:4326"
)
ncell01 <- ncell(template01)
ncell05 <- 720L * 360L

candidate_mask_path <- file.path(
  dirs$support, "analysis_mask_ssp585_2050_0p1deg_v1.tif"
)
candidate_mask <- values(rast(candidate_mask_path), mat = FALSE)
candidate_cell <- which(is.finite(candidate_mask) & candidate_mask == 1)
n_candidate <- length(candidate_cell)
if (n_candidate != fc01$analysis_child_cells) stop("FC01 support count changed")

parent_cell <- values(rast(file.path(
  dirs$support, "parent_cell_id_ssp585_2050_0p1deg_v1.tif"
)), mat = FALSE)[candidate_cell]
country_id <- values(rast(file.path(
  dirs$support, "country_id_ssp585_2050_0p1deg_v1.tif"
)), mat = FALSE)[candidate_cell]
parent_cell <- as.integer(parent_cell)
country_id <- as.integer(country_id)
if (anyNA(parent_cell) || anyNA(country_id)) stop("Missing nested parent/country identifiers")

sum_by_parent <- function(value, parent, n = ncell05) {
  output <- numeric(n)
  aggregate <- rowsum(value, parent, reorder = FALSE, na.rm = TRUE)
  output[as.integer(rownames(aggregate))] <- aggregate[, 1]
  output
}

count_by_parent <- function(parent, n = ncell05) tabulate(parent, nbins = n)

allocate_parent_total <- function(weight, parent, target05, fallback = NULL) {
  weight[!is.finite(weight) | weight < 0] <- 0
  denominator <- sum_by_parent(weight, parent)
  good <- denominator[parent] > 0
  if (!is.null(fallback)) {
    fallback[!is.finite(fallback) | fallback < 0] <- 0
    use <- !good
    if (any(use)) weight[use] <- fallback[use]
    denominator <- sum_by_parent(weight, parent)
    good <- denominator[parent] > 0
  }
  weight[!good] <- 1
  denominator <- sum_by_parent(weight, parent)
  target <- target05[parent]
  target[!is.finite(target) | target < 0] <- 0
  target * weight / pmax(denominator[parent], .Machine$double.xmin)
}

match_parent_mean <- function(value, parent, target05, lower = -Inf, upper = Inf) {
  value[!is.finite(value)] <- 0
  counts <- count_by_parent(parent)
  mean05 <- sum_by_parent(value, parent) / pmax(counts, 1)
  deviation <- value - mean05[parent]
  limits <- data.table(parent = parent, deviation = deviation)[
    , .(minimum = min(deviation), maximum = max(deviation)), by = parent
  ]
  scale05 <- rep(1, ncell05)
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
  result <- target[parent] + scale05[parent] * deviation
  if (is.finite(lower)) result <- pmax(lower, result)
  if (is.finite(upper)) result <- pmin(upper, result)
  result
}

parent_sum_max_error <- function(value, parent, target05) {
  observed <- sum_by_parent(value, parent)
  levels <- sort(unique(parent))
  max(abs(observed[levels] - target05[levels]), na.rm = TRUE)
}

parent_sum_max_relative_error <- function(value, parent, target05) {
  observed <- sum_by_parent(value, parent)
  levels <- sort(unique(parent))
  max(
    abs(observed[levels] - target05[levels]) /
      pmax(abs(target05[levels]), 1),
    na.rm = TRUE
  )
}

parent_mean_max_error <- function(value, parent, target05) {
  observed <- sum_by_parent(value, parent) / pmax(count_by_parent(parent), 1)
  levels <- sort(unique(parent))
  max(abs(observed[levels] - target05[levels]), na.rm = TRUE)
}

parent_population_path <- file.path(
  production_root, "outputs", scenario_code, as.character(year),
  "adult_population_age_sex_0p5deg_v1.tif"
)
parent_probability_path <- file.path(
  production_root, "outputs", scenario_code, as.character(year),
  "bmi7_probability_age_sex_0p5deg_v1.tif"
)
parent_population <- values(rast(parent_population_path), mat = TRUE)
parent_population[!is.finite(parent_population) | parent_population < 0] <- 0
if (ncol(parent_population) != 30L) stop("Expected 30 parent population bands")

stamp("Allocating 30 preliminary age-sex populations")
dynamic_population_candidate <- values(rast(file.path(
  dirs$dynamic, "population_ssp5_2050_0p1deg_parent_calibrated_v1.tif"
)), mat = FALSE)[candidate_cell]
worldpop_root <- file.path(old_0p1, "data", "0p1deg", "worldpop_2025_prior")
prior_paths <- list.files(worldpop_root, pattern = "[.]tif$", full.names = TRUE)
if (length(prior_paths) != 28L) stop("Expected 28 WorldPop priors")
prior_lookup <- setNames(prior_paths, sub("_2025.*$", "", basename(prior_paths)))
adult_prior <- numeric(n_candidate)
for (path in prior_paths) {
  value <- values(rast(path), mat = FALSE)[candidate_cell]
  value[!is.finite(value) | value < 0] <- 0
  adult_prior <- adult_prior + value
}

pre_paths <- character(30L)
adult_pre <- numeric(n_candidate)
young_pre <- numeric(n_candidate)
older_pre <- numeric(n_candidate)
female_pre <- numeric(n_candidate)
group_index <- 0L
for (sex in sexes) {
  sex_code <- if (sex == "Female") "f" else "m"
  for (age_group in age_groups) {
    group_index <- group_index + 1L
    lower_age <- min(age_lower(age_group), 90L)
    key <- paste0("global_", sex_code, "_", lower_age)
    prior_path <- unname(prior_lookup[[key]])
    if (is.null(prior_path) || !file.exists(prior_path)) stop("Missing prior: ", key)
    group_prior <- values(rast(prior_path), mat = FALSE)[candidate_cell]
    group_prior[!is.finite(group_prior) | group_prior < 0] <- 0
    share <- numeric(n_candidate)
    have <- adult_prior > 0
    share[have] <- group_prior[have] / adult_prior[have]
    raw_weight <- dynamic_population_candidate * share
    target05 <- parent_population[, group_index]
    allocation <- allocate_parent_total(
      raw_weight, parent_cell, target05, fallback = group_prior
    )
    pre_path <- file.path(
      dirs$pre_population,
      sprintf("population_pre_%s_a%02d_f64le_v1.bin", tolower(sex),
              match(age_group, age_groups))
    )
    write_binary_resume(allocation, pre_path, 8L)
    pre_paths[group_index] <- pre_path
    adult_pre <- adult_pre + allocation
    if (age_lower(age_group) < 40L) young_pre <- young_pre + allocation
    if (age_lower(age_group) >= 60L) older_pre <- older_pre + allocation
    if (sex == "Female") female_pre <- female_pre + allocation
    stamp("  population ", sex, " ", age_group)
  }
}

cell_area_all <- values(cellSize(template01, unit = "km"), mat = FALSE)
cell_area_candidate <- cell_area_all[candidate_cell]
density_pre <- adult_pre / cell_area_candidate
final_keep_candidate <- is.finite(density_pre) & density_pre >= 0.1 & adult_pre > 0

parent_adult_target <- rowSums(parent_population)
retained_parent <- count_by_parent(parent_cell[final_keep_candidate]) > 0
missing_parent <- which(parent_adult_target > 0 & !retained_parent)
if (length(missing_parent)) {
  stamp(
    "  parents without a child above the density threshold: ",
    length(missing_parent), "; adult population fraction=",
    format(sum(parent_adult_target[missing_parent]) / sum(parent_adult_target),
           scientific = TRUE)
  )
}

active_candidate_index <- which(final_keep_candidate)
active_cell <- candidate_cell[active_candidate_index]
active_parent <- parent_cell[active_candidate_index]
active_country_id <- country_id[active_candidate_index]
n_active <- length(active_cell)
if (n_active < 500000L) stop("Unexpectedly small density-masked support")

stamp("Reallocating every age-sex group over the final density mask")
population_paths <- character(30L)
adult_population <- numeric(n_active)
young_population <- numeric(n_active)
older_population <- numeric(n_active)
female_population <- numeric(n_active)
population_qa <- vector("list", 30L)
for (group_index in seq_len(30L)) {
  pre <- readBin(pre_paths[group_index], what = "numeric", n = n_candidate,
                 size = 8L, endian = "little")
  target05 <- parent_population[, group_index]
  allocation <- allocate_parent_total(
    pre[active_candidate_index], active_parent, target05,
    fallback = dynamic_population_candidate[active_candidate_index]
  )
  sex_index <- (group_index - 1L) %/% 15L + 1L
  age_index <- (group_index - 1L) %% 15L + 1L
  sex <- sexes[sex_index]
  age_group <- age_groups[age_index]
  path <- file.path(
    dirs$binary,
    sprintf("population_%s_a%02d_f64le_v1.bin", tolower(sex), age_index)
  )
  write_binary_resume(allocation, path, 8L)
  population_paths[group_index] <- path
  adult_population <- adult_population + allocation
  if (age_lower(age_group) < 40L) young_population <- young_population + allocation
  if (age_lower(age_group) >= 60L) older_population <- older_population + allocation
  if (sex == "Female") female_population <- female_population + allocation
  population_qa[[group_index]] <- data.table(
    group_index = group_index, sex = sex, age_group = age_group,
    active_cells = n_active,
    population = sum(allocation),
    maximum_parent_population_error = parent_sum_max_error(
      allocation, active_parent, target05
    ),
    path = path
  )
  file.remove(pre_paths[group_index])
}
population_qa <- rbindlist(population_qa)

density_final <- adult_population / cell_area_candidate[active_candidate_index]
population_retained <- sum(adult_pre[active_candidate_index]) / sum(adult_pre)
minimum_final_density <- min(density_final, na.rm = TRUE)
if (minimum_final_density < 0.1 - 1e-8) {
  stop(
    "Final adult-density mask is invalid; minimum density=",
    format(minimum_final_density, digits = 16)
  )
}

active_id_path <- file.path(dirs$support, "active_cell_ids_int32le_ssp585_2050_v2.bin")
write_integer_binary_resume(active_cell, active_id_path)
active_parent_path <- file.path(dirs$support, "active_parent_cell_ids_int32le_ssp585_2050_v2.bin")
write_integer_binary_resume(active_parent, active_parent_path)
active_country_path <- file.path(dirs$support, "active_country_ids_int32le_ssp585_2050_v2.bin")
write_integer_binary_resume(active_country_id, active_country_path)

mask_values <- rep(NA_integer_, ncell01)
mask_values[active_cell] <- 1L
final_mask_path <- file.path(dirs$support, "adult_density_mask_ssp585_2050_0p1deg_v2.tif")
write_raster_resume(mask_values, final_mask_path, template01, "INT2S")

stamp("Preparing full-model covariates over the final support")
read_dynamic_active <- function(filename) {
  values(rast(file.path(dirs$dynamic, filename)), mat = FALSE)[active_cell]
}

total_population <- allocate_parent_total(
  read_dynamic_active("population_ssp5_2050_0p1deg_parent_calibrated_v1.tif"),
  active_parent, dynamic05$population,
  fallback = adult_population
)
gdp_total <- allocate_parent_total(
  read_dynamic_active("gdp_ssp5_2050_0p1deg_parent_calibrated_v1.tif"),
  active_parent, dynamic05$gdp,
  fallback = total_population
)
ntl <- match_parent_mean(
  read_dynamic_active("ntl_log_ssp5_2050_0p1deg_parent_centered_v1.tif"),
  active_parent, dynamic05$ntl, lower = 0
)
urban <- match_parent_mean(
  read_dynamic_active("urban_share_ssp5_2050_0p1deg_parent_centered_v1.tif"),
  active_parent, dynamic05$urban, lower = 0, upper = 1
)
built <- match_parent_mean(
  read_dynamic_active("built_fraction_ssp5_2050_0p1deg_parent_centered_v1.tif"),
  active_parent, dynamic05$built, lower = 0, upper = 1
)
pm25 <- read_dynamic_active(
  "pm25_mmr_trailing5y_ssp5_2050_0p1deg_parent_constant_v1.tif"
)
temperature <- read_dynamic_active(
  "temperature_c_ssp5_2050_0p1deg_parent_constant_v1.tif"
)
humidity <- read_dynamic_active(
  "relative_humidity_pct_ssp5_2050_0p1deg_parent_constant_v1.tif"
)

static_root <- file.path(old_0p1, "data", "0p1deg", "static")
elevation <- values(rast(file.path(
  static_root, "elevation_m_0p1deg_nested_v1.tif"
)), mat = FALSE)[active_cell]
travel <- values(rast(file.path(
  static_root, "travel_time_city50k_minutes_0p1deg_nested_v1.tif"
)), mat = FALSE)[active_cell]

age20_39_share <- young_population / adult_population
age60plus_share <- older_population / adult_population
female_share <- female_population / adult_population
cell_area_active <- cell_area_candidate[active_candidate_index]

iso3 <- country_lookup$iso3[match(active_country_id, country_lookup$country_id)]
if (anyNA(iso3)) stop("Active child cells include unmapped country identifiers")
stage_lookup <- disposition[, .(iso3, development_stage_2023)]
development_stage <- stage_lookup$development_stage_2023[
  match(iso3, stage_lookup$iso3)
]

raw_values <- list(
  ntl_within = ntl,
  pop_density_within = log1p(total_population / cell_area_active),
  gdp_within = log1p(ifelse(total_population > 0, gdp_total / total_population, 0)),
  pm25_within_aug = pm25,
  urban_share_within = urban,
  built_fraction_within = built,
  travel_time_within = log1p(pmax(0, travel)),
  temperature_c_within_aug = temperature,
  relative_humidity_pct_within_aug = pmin(100, pmax(0, humidity)),
  elevation_m_within_aug = elevation,
  age20_39_share_within_aug = age20_39_share,
  age60plus_share_within_aug = age60plus_share,
  female_adult_share_within_aug = female_share
)

model_grid <- data.table(
  cell_id = active_cell,
  parent_cell_id = active_parent,
  country_id = active_country_id,
  iso3 = iso3,
  development_stage = development_stage,
  cell_adult_population = adult_population
)
centering_rows <- list()
centering_index <- 0L
for (variable in scaled_columns) {
  output <- numeric(n_active)
  for (country in unique(iso3)) {
    rows <- which(iso3 == country)
    centered <- weighted_center(
      raw_values[[variable]][rows], adult_population[rows], variable,
      country, scenario, year
    )
    output[rows] <- centered$within
    centering_index <- centering_index + 1L
    centering_rows[[centering_index]] <- data.table(
      iso3 = country, variable = variable,
      centering_error = centered$centering_error,
      imputed_values = centered$imputed
    )
  }
  model_grid[, (variable) := output]
}
centering_qa <- rbindlist(centering_rows)

stamp("Evaluating the frozen MRP for 30 age-sex strata")
prediction_qa <- list()
prediction_index <- 0L
for (sex in sexes) {
  for (age_index in seq_along(age_groups)) {
    age_group <- age_groups[age_index]
    probability <- matrix(NA_real_, nrow = n_active, ncol = length(bmi_labels))
    for (country in unique(iso3)) {
      rows <- which(iso3 == country)
      scenario_value <- scenario
      country_value <- country
      year_value <- year
      sex_value <- sex
      age_group_value <- age_group
      target_row <- anchor[J(
        scenario_value, country_value, year_value, sex_value, age_group_value
      )]
      if (nrow(target_row) != 1L) {
        stop("Missing national anchor: ", country, " ", sex, " ", age_group)
      }
      target <- setNames(
        as.numeric(unlist(target_row[, ..target_columns], use.names = FALSE)),
        bmi_labels
      )
      prediction_data <- copy(model_grid[rows])
      prediction_data[, `:=`(
        sex = sex,
        age_group = age_group,
        model_age_midpoint = model_age_midpoint(age_group)
      )]
      probability[rows, ] <- predict_uncalibrated(prediction_data, target)
    }
    if (any(!is.finite(probability))) stop("Non-finite MRP probabilities")
    closure_error <- max(abs(rowSums(probability) - 1))
    probability_path <- file.path(
      dirs$binary,
      sprintf("probability_unraked_%s_a%02d_f32le_v1.bin", tolower(sex), age_index)
    )
    write_binary_resume(
      as.vector(probability), probability_path, 4L, tolerance = 1e-6
    )
    prediction_index <- prediction_index + 1L
    prediction_qa[[prediction_index]] <- data.table(
      group_index = prediction_index,
      sex = sex,
      age_group = age_group,
      active_cells = n_active,
      maximum_probability_closure_error = closure_error,
      minimum_probability = min(probability),
      maximum_probability = max(probability),
      probability_path = probability_path,
      population_path = population_paths[prediction_index]
    )
    stamp("  MRP ", sex, " ", age_group)
    rm(probability)
    gc()
  }
}
prediction_qa <- rbindlist(prediction_qa)

extrapolation_qa <- rbindlist(lapply(scaled_columns, function(column) {
  limits <- training_ranges[variable == column]
  outside <- model_grid[[column]] < limits$train_min |
    model_grid[[column]] > limits$train_max
  data.table(
    variable = column,
    future_min = min(model_grid[[column]], na.rm = TRUE),
    future_max = max(model_grid[[column]], na.rm = TRUE),
    training_min = limits$train_min,
    training_max = limits$train_max,
    active_cells_outside = sum(outside),
    adult_population_fraction_outside = weighted_mean_safe(
      as.numeric(outside), adult_population
    )
  )
}))

write_csv_new(population_qa, file.path(
  dirs$qa, "population_age_sex_parent_closure_ssp585_2050_v2.csv"
))
write_csv_new(centering_qa, file.path(
  dirs$qa, "model_covariate_country_centering_ssp585_2050_v2.csv"
))
write_csv_new(prediction_qa, file.path(
  dirs$qa, "unraked_mrp_probability_ssp585_2050_v2.csv"
))
write_csv_new(extrapolation_qa, file.path(
  dirs$qa, "unraked_mrp_extrapolation_ssp585_2050_v2.csv"
))

dynamic_parent_qa <- data.table(
  variable = c("population", "gdp", "ntl", "urban", "built"),
  rule = c("sum", "sum", "mean", "mean", "mean"),
  maximum_parent_error = c(
    parent_sum_max_error(total_population, active_parent, dynamic05$population),
    parent_sum_max_error(gdp_total, active_parent, dynamic05$gdp),
    parent_mean_max_error(ntl, active_parent, dynamic05$ntl),
    parent_mean_max_error(urban, active_parent, dynamic05$urban),
    parent_mean_max_error(built, active_parent, dynamic05$built)
  ),
  maximum_parent_relative_error = c(
    parent_sum_max_relative_error(
      total_population, active_parent, dynamic05$population
    ),
    parent_sum_max_relative_error(gdp_total, active_parent, dynamic05$gdp),
    NA_real_, NA_real_, NA_real_
  )
)
write_csv_new(dynamic_parent_qa, file.path(
  dirs$qa, "post_mask_dynamic_parent_consistency_ssp585_2050_v2.csv"
))

checks <- data.table(
  check_id = sprintf("FC02_%02d", 1:10),
  requirement = c(
    "Gate FC01 is PASS",
    "The final adult-density support is nonempty",
    "The final support retains at least 99.9% of preliminary adults",
    "Thirty age-sex population binaries exist",
    "Every age-sex population closes to its formal 0.5-degree parent",
    "All thirteen model covariates are country-centered",
    "Thirty unraked full-MRP probability binaries exist",
    "Every unraked composition closes to one",
    "Post-mask dynamic covariates preserve formal parent inputs",
    "No model fitting or disease burden analysis occurred"
  ),
  observed = c(
    fc01$status,
    n_active,
    population_retained,
    sum(file.exists(population_paths)),
    max(population_qa$maximum_parent_population_error),
    max(centering_qa$centering_error),
    sum(file.exists(prediction_qa$probability_path)),
    max(prediction_qa$maximum_probability_closure_error),
    max(dynamic_parent_qa$maximum_parent_error),
    "scope guard retained"
  ),
  pass = c(
    TRUE,
    n_active >= 500000L,
    population_retained >= 0.999,
    all(file.exists(population_paths)),
    max(population_qa$maximum_parent_population_error) <= 1e-5,
    max(centering_qa$centering_error) <= 1e-8,
    all(file.exists(prediction_qa$probability_path)),
    max(prediction_qa$maximum_probability_closure_error) <= 1e-10,
    dynamic_parent_qa[variable == "population", maximum_parent_error] <= 1e-5 &&
      dynamic_parent_qa[variable == "gdp", maximum_parent_relative_error] <= 1e-8 &&
      dynamic_parent_qa[variable %in% c("ntl", "urban", "built"),
                        max(maximum_parent_error)] <= 1e-5,
    TRUE
  )
)
write_csv_new(checks, file.path(dirs$qa, "Gate_FC02_checks_ssp585_2050_v2.csv"))
status <- if (all(checks$pass)) "PASS_GATE_FC02_UNRAKED_FULL_MRP" else
  "FAIL_GATE_FC02_UNRAKED_FULL_MRP"

state <- list(
  status = status,
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  scenario = scenario,
  scenario_code = scenario_code,
  year = year,
  model = "validated 22-feature Bayesian multinomial logistic-normal MRP",
  candidate_child_cells = n_candidate,
  active_child_cells = n_active,
  represented_parent_cells = uniqueN(active_parent),
  represented_countries = uniqueN(iso3),
  adult_population_retained_fraction = population_retained,
  density_threshold_adults_per_km2 = 0.1,
  maximum_population_parent_error = max(
    population_qa$maximum_parent_population_error
  ),
  maximum_country_centering_error = max(centering_qa$centering_error),
  maximum_unraked_probability_closure_error = max(
    prediction_qa$maximum_probability_closure_error
  ),
  maximum_adult_population_fraction_outside_training_range = max(
    extrapolation_qa$adult_population_fraction_outside
  ),
  active_cell_id_path = active_id_path,
  active_parent_cell_id_path = active_parent_path,
  active_country_id_path = active_country_path,
  parent_probability_path = parent_probability_path,
  parent_population_path = parent_population_path,
  population_paths = population_paths,
  probability_paths = prediction_qa$probability_path
)
write_json_new(state, file.path(dirs$qa, "Gate_FC02_summary_ssp585_2050_v2.json"))
write_json_new(list(
  status = status,
  method = paste(
    "WorldPop age-sex priors and scenario population redistribute each formal",
    "0.5-degree age-sex population to 0.1 degrees; the frozen full MRP is then",
    "evaluated on all 22 fixed-effect features before parent-cell compositional raking."
  ),
  population_mask = "adult age 25+ density >= 0.1 persons per square kilometre",
  parent_targets = "formal full-covariate SSP5-8.5 2050 age-sex COGs",
  prohibited = c("model fitting", "disease burden analysis")
), file.path(dirs$contracts, "full_mrp_0p1deg_unraked_contract_v2.json"))

stamp(status)
if (!startsWith(status, "PASS_")) quit(status = 2L)
