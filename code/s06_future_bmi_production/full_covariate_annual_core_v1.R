if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

suppressPackageStartupMessages({
  library(data.table)
  library(digest)
  library(jsonlite)
  library(rstan)
  library(splines)
  library(terra)
})

rstan_options(auto_write = FALSE)
options(mc.cores = 1L)

project_root <- file.path(.cfg$RESULTS)
root <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "full_covariate_annual_0p5deg_production_v1_20260906"
)
model_root <- file.path(
  project_root, "04_future_capable_mrp",
  "full_covariate_mrp_refit_v1_20260905"
)
gate_c <- file.path(
  project_root, "02_harmonized_covariates",
  "gate_c_v1_country_coverage_remediated_20260828_122619"
)
full_covariates <- file.path(
  project_root, "02_harmonized_covariates",
  "full_covariate_extension_v1_20260905"
)
ntl_root <- file.path(
  project_root, "02_harmonized_covariates", "ntl_bridge1p5_v1_20260906"
)
preflight <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "full_covariate_global_preflight_v2_20260906"
)
gdp_remediation <- file.path(
  project_root, "02_harmonized_covariates",
  "gdp_spatial_coverage_remediation_v1_20260906"
)
old_preflight <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "phase6_global_ssp245_contract_preflight_v3_20260830"
)
input_remediation <- file.path(
  project_root, "06_future_bmi_2025_2050",
  "phase6_global_ssp245_input_remediation_v4_20260830"
)

training_path <- file.path(model_root, "data", "bmi7_full_covariate_training_v1.csv.gz")
fit_path <- file.path(
  model_root, "model_fit", "models", "checkpoints", "full_model_fit_v1.rds"
)
recipe_path <- file.path(
  model_root, "model_fit", "models", "checkpoints", "full_model_recipe_v1.rds"
)
anchor_path <- file.path(
  preflight, "inputs", "national_bmi7_scenario_anchors_full204_v2.csv.gz"
)
country_raster_path <- file.path(
  input_remediation, "data", "0p5deg",
  "country_id_0p5deg_phase6_remediated_v1.tif"
)
country_lookup_path <- file.path(
  input_remediation, "data", "0p5deg",
  "country_id_lookup_phase6_remediated_v1.csv"
)
disposition_path <- file.path(
  old_preflight, "tables", "Phase6_global_country_disposition_v3.csv"
)

scenarios <- c(
  "SSP1-2.6" = "ssp1",
  "SSP2-4.5" = "ssp2",
  "SSP3-7.0" = "ssp3",
  "SSP5-8.5" = "ssp5"
)
production_years <- 2025:2050
spatial_countries <- fread(disposition_path)[
  disposition == "SPATIAL_READY_DIRECT", sort(unique(iso3))
]

bmi_labels <- c(
  "bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
  "bmi_30_35", "bmi_35_40", "bmi_ge_40"
)
nonreference_labels <- bmi_labels[bmi_labels != "bmi_20_25"]
target_columns <- paste0("p_", bmi_labels)
support <- c(17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5)
age_groups <- c(
  "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
  "55-59", "60-64", "65-69", "70-74", "75-79", "80-84",
  "85-89", "90-94", "95+"
)
sexes <- c("Female", "Male")
scaled_columns <- c(
  "ntl_within", "pop_density_within", "gdp_within", "pm25_within_aug",
  "urban_share_within", "built_fraction_within", "travel_time_within",
  "temperature_c_within_aug", "relative_humidity_pct_within_aug",
  "elevation_m_within_aug", "age20_39_share_within_aug",
  "age60plus_share_within_aug", "female_adult_share_within_aug"
)

# Frozen before annual production.
limits <- list(
  maximum_national_anchor_error_pp = 0.01,
  maximum_probability_closure_error = 1e-10,
  maximum_centering_error = 1e-8,
  minimum_country_scenario_year_population_retained = 0.98,
  maximum_2025_population_weighted_mean_w2_kg_m2 = 0.05,
  maximum_2025_population_weighted_mean_absolute_obesity_difference_pp = 0.10
)

for (directory in c(
  "contracts", "data", "logs", "outputs", "qa", "reports", "checksums",
  "checkpoints"
)) {
  dir.create(file.path(root, directory), recursive = TRUE, showWarnings = FALSE)
}

timestamp_utc <- function() format(Sys.time(), tz = "UTC", usetz = TRUE)
sha256_file <- function(path) digest(path, algo = "sha256", file = TRUE, serialize = FALSE)

write_new_csv <- function(value, path, compress = grepl("[.]gz$", path)) {
  if (file.exists(path)) stop("Refusing to overwrite output: ", path, call. = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  fwrite(value, path, compress = if (compress) "gzip" else "none", bom = !compress)
}

write_new_json <- function(value, path) {
  if (file.exists(path)) stop("Refusing to overwrite output: ", path, call. = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_json(value, path, pretty = TRUE, auto_unbox = TRUE, na = "null")
}

write_new_text <- function(value, path) {
  if (file.exists(path)) stop("Refusing to overwrite output: ", path, call. = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(value, path, useBytes = TRUE)
}

weighted_mean_safe <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  if (!any(keep)) return(NA_real_)
  sum(value[keep] * weight[keep]) / sum(weight[keep])
}

weighted_quantile <- function(value, weight, probability) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  if (!any(keep)) return(NA_real_)
  value <- value[keep]
  weight <- weight[keep]
  order_index <- order(value)
  value <- value[order_index]
  weight <- weight[order_index]
  value[which(cumsum(weight) / sum(weight) >= probability)[1L]]
}

weighted_center <- function(value, weight, variable, iso3, scenario, year) {
  center <- weighted_mean_safe(value, weight)
  if (!is.finite(center)) {
    stop("No finite values for ", variable, " in ", iso3, " ", scenario, " ", year)
  }
  imputed <- !is.finite(value)
  value[imputed] <- center
  within <- value - weighted_mean_safe(value, weight)
  list(
    within = within,
    imputed = sum(imputed),
    centering_error = abs(weighted_mean_safe(within, weight))
  )
}

close_composition <- function(value) {
  value <- pmax(as.matrix(value), 1e-12)
  value / rowSums(value)
}

rake_composition <- function(probability, target, weight, tolerance = 1e-12,
                             max_iterations = 500L) {
  probability <- close_composition(probability)
  target <- pmax(as.numeric(target), 1e-12)
  target <- target / sum(target)
  normalized_weight <- as.numeric(weight) / sum(weight)
  error <- Inf
  iteration <- 0L
  for (iteration in seq_len(max_iterations)) {
    current <- colSums(probability * normalized_weight)
    probability <- sweep(probability, 2L, target / pmax(current, 1e-15), "*")
    probability <- probability / rowSums(probability)
    error <- max(abs(colSums(probability * normalized_weight) - target))
    if (error <= tolerance) break
  }
  if (error > tolerance) stop("Raking did not converge; error=", error)
  list(probability = probability, iterations = iteration, error = error)
}

row_wasserstein2 <- function(reference, comparison) {
  reference <- close_composition(reference)
  comparison <- close_composition(comparison)
  output <- numeric(nrow(reference))
  for (row_index in seq_len(nrow(reference))) {
    boundaries <- sort(unique(c(
      0, 1, cumsum(reference[row_index, ]), cumsum(comparison[row_index, ])
    )))
    boundaries <- pmin(1, pmax(0, boundaries))
    middle <- (head(boundaries, -1L) + tail(boundaries, -1L)) / 2
    width <- diff(boundaries)
    reference_index <- findInterval(
      middle, c(0, cumsum(reference[row_index, ])),
      rightmost.closed = TRUE, all.inside = TRUE
    )
    comparison_index <- findInterval(
      middle, c(0, cumsum(comparison[row_index, ])),
      rightmost.closed = TRUE, all.inside = TRUE
    )
    reference_index <- pmin(length(support), pmax(1L, reference_index))
    comparison_index <- pmin(length(support), pmax(1L, comparison_index))
    output[row_index] <- sqrt(sum(
      width * (support[reference_index] - support[comparison_index])^2
    ))
  }
  output
}

posterior_mean <- function(draws, name) {
  value <- draws[[name]]
  apply(value, seq.int(2L, length(dim(value))), mean)
}

normalize_anchor_age <- function(value) {
  value <- gsub(" to ", "-", value, fixed = TRUE)
  sub("95 plus", "95+", value, fixed = TRUE)
}

age_lower <- function(value) as.integer(sub("[^0-9].*$", "", value))
model_age_midpoint <- function(value) ifelse(age_lower(value) >= 85L, 87, age_lower(value) + 2)

required_inputs <- c(
  training_path, fit_path, recipe_path, anchor_path, country_raster_path,
  country_lookup_path, disposition_path,
  file.path(gate_c, "data", "0p5deg", "static", "elevation_m_0p5deg.tif"),
  file.path(gate_c, "data", "0p5deg", "static", "travel_time_city50k_minutes_0p5deg.tif")
)
if (any(!file.exists(required_inputs))) {
  stop("Missing required input: ", paste(required_inputs[!file.exists(required_inputs)], collapse = "; "))
}

contract <- list(
  phase = "Full-covariate annual 0.5-degree production",
  version = "v1",
  status_before_run = "AUTHORIZED_BY_GLOBAL_KEY_YEAR_PILOT_V1",
  created_at_utc = timestamp_utc(),
  model = "Validated 22-feature Bayesian multinomial logistic-normal MRP",
  years = production_years,
  scenarios = names(scenarios),
  spatial_countries = spatial_countries,
  national_only_locations = disposition_path,
  national_anchor = paste(
    "SSP2 uses the GBD 2021 broad-category-constrained seven-category anchor;",
    "SSP1/3/5 use the validated Phase 6J national ALR contrasts from the common 2025 anchor"
  ),
  nighttime_light_bridge_slope = 1.5,
  dynamic_covariates = c(
    "population", "GDP", "nighttime light", "PM2.5 mass mixing ratio",
    "urban share", "built fraction proxy", "temperature", "relative humidity"
  ),
  population_allocation = paste(
    "Scenario-specific grid population redistributes the WorldPop 2025 age-sex prior;",
    "each national age-sex group is then calibrated to the matching IIASA WiC total"
  ),
  population_mask = "Scenario-year adult age 25+ density >= 0.1 persons per km2",
  scenario_branch_bridge = paste(
    "All scenarios share the SSP2 2025 gridded covariate and national age-sex",
    "population baseline; each future path applies its change from its own 2025 node"
  ),
  output_contract = list(
    grid = "global WGS84 EPSG:4326 at 0.5 degree",
    datatype = "Float32",
    format = "Cloud Optimized GeoTIFF with ZSTD compression",
    nodata = -9999,
    probability_groups = c("overall", "sex", "age", "age_sex"),
    probability_bands_per_stratum = bmi_labels,
    companion_population_groups = c("overall", "sex", "age", "age_sex"),
    checkpoint_unit = "scenario-year"
  ),
  prespecified_limits = limits,
  formal_annual_production_authorized = TRUE,
  disease_burden_analysis_authorized = FALSE,
  gdp_policy = paste(
    "A1-calibrated GDP where available; otherwise Wang-Sun A4 is used only",
    "as a country-centered within-country spatial allocation pattern"
  ),
  input_paths = c(required_inputs, gdp_remediation)
)
contract_path <- file.path(root, "contracts", "annual_0p5deg_production_contract_frozen_v1.json")
if (!file.exists(contract_path)) write_new_json(contract, contract_path)

country_raster <- rast(country_raster_path)
country_lookup <- fread(country_lookup_path)
country_values <- values(country_raster, mat = FALSE)
cell_area <- values(cellSize(country_raster, unit = "km"), mat = FALSE)
cell_xy <- xyFromCell(country_raster, seq_len(ncell(country_raster)))

disposition <- fread(disposition_path)
missing_countries <- setdiff(spatial_countries, disposition$iso3)
if (length(missing_countries)) {
  stop("Spatial countries absent from disposition: ", paste(missing_countries, collapse = ","))
}
not_spatial <- disposition[
  iso3 %in% spatial_countries & disposition != "SPATIAL_READY_DIRECT", iso3
]
if (length(not_spatial)) {
  stop("Production countries not spatial-ready: ", paste(not_spatial, collapse = ","))
}

anchor_all <- fread(anchor_path)
anchor_all <- anchor_all[scenario %in% names(scenarios) & year %in% production_years]
anchor_all[, age_group := normalize_anchor_age(age_group_output)]
# Put all demographic pathways on the same SSP2 2025 base, then preserve each
# pathway's age-sex-specific growth ratio. This prevents an artificial jump at
# the scenario branch while retaining scenario-specific future populations.
scenario_population_2025 <- anchor_all[year == 2025, .(
  scenario, iso3, sex, age_group,
  scenario_population_2025 = population_persons
)]
ssp2_population_2025 <- anchor_all[scenario == "SSP2-4.5" & year == 2025, .(
  iso3, sex, age_group,
  common_population_2025 = population_persons
)]
anchor_all[, population_persons_unbridged := population_persons]
anchor_all[scenario_population_2025,
  on = .(scenario, iso3, sex, age_group),
  scenario_population_2025 := i.scenario_population_2025
]
anchor_all[ssp2_population_2025,
  on = .(iso3, sex, age_group),
  common_population_2025 := i.common_population_2025
]
anchor_all[, population_persons := common_population_2025 *
  population_persons_unbridged / scenario_population_2025]
national_anchor_path <- file.path(
  root, "data", "national_bmi7_anchors_and_population_all204_2025_2050_v1.csv.gz"
)
if (!file.exists(national_anchor_path)) write_new_csv(anchor_all, national_anchor_path)
national_only_path <- file.path(root, "data", "national_only_no_0p5_cell_v1.csv")
if (!file.exists(national_only_path)) write_new_csv(
  disposition[disposition == "NATIONAL_ONLY_NO_0P5_CELL"],
  national_only_path, compress = FALSE
)

anchor <- anchor_all[iso3 %in% spatial_countries]
setkey(anchor, scenario, iso3, year, sex, age_group)
expected_anchor_rows <- length(scenarios) * length(spatial_countries) *
  length(production_years) * length(sexes) * length(age_groups)
if (nrow(anchor) != expected_anchor_rows) stop("Incomplete production anchors")
if (any(!is.finite(anchor$population_persons)) || any(anchor$population_persons < 0)) {
  stop("Invalid production national age-sex population")
}

bundle <- readRDS(recipe_path)
recipe <- bundle$recipe
expected_features <- c(
  "intercept", "ntl_ns1", "ntl_ns2", "ntl_ns3",
  "pop_ns1", "pop_ns2", "pop_ns3", "gdp_within", "pm25_within_aug",
  "urban_share_within", "built_fraction_within", "travel_time_within",
  "temperature_c_within_aug", "relative_humidity_pct_within_aug",
  "elevation_m_within_aug", "age20_39_share_within_aug",
  "age60plus_share_within_aug", "female_adult_share_within_aug",
  "ntl_by_female", "pop_by_female", "ntl_by_age", "pop_by_age"
)
if (!identical(bundle$feature_names, expected_features)) {
  stop("Unexpected full-model feature contract")
}

fit <- readRDS(fit_path)
posterior_names <- c(
  "beta", "z_country_intercept", "z_country_ntl", "z_country_pop",
  "sigma_country_intercept", "sigma_country_ntl", "sigma_country_pop",
  "z_stage_ntl", "z_stage_pop", "sigma_stage_ntl", "sigma_stage_pop"
)
draws <- rstan::extract(fit, pars = posterior_names, permuted = TRUE)
beta <- posterior_mean(draws, "beta")
z_country_intercept <- posterior_mean(draws, "z_country_intercept")
z_country_ntl <- posterior_mean(draws, "z_country_ntl")
z_country_pop <- posterior_mean(draws, "z_country_pop")
z_stage_ntl <- posterior_mean(draws, "z_stage_ntl")
z_stage_pop <- posterior_mean(draws, "z_stage_pop")
sigma_country_intercept <- colMeans(draws$sigma_country_intercept)
sigma_country_ntl <- colMeans(draws$sigma_country_ntl)
sigma_country_pop <- colMeans(draws$sigma_country_pop)
sigma_stage_ntl <- colMeans(draws$sigma_stage_ntl)
sigma_stage_pop <- colMeans(draws$sigma_stage_pop)
rm(draws, fit)
gc()

scale_value <- function(value, column) {
  (value - recipe$scaling[[column]]$center) / recipe$scaling[[column]]$scale
}

make_design <- function(data) {
  scaled <- lapply(scaled_columns, function(column) scale_value(data[[column]], column))
  names(scaled) <- scaled_columns
  ntl_basis <- predict(recipe$ntl_spline, newx = scaled$ntl_within)
  pop_basis <- predict(recipe$pop_spline, newx = scaled$pop_density_within)
  age_scaled <- (data$model_age_midpoint - recipe$age_scaling$center) /
    recipe$age_scaling$scale
  female <- as.numeric(data$sex == "Female")
  X <- cbind(
    intercept = 1,
    ntl_ns1 = ntl_basis[, 1], ntl_ns2 = ntl_basis[, 2], ntl_ns3 = ntl_basis[, 3],
    pop_ns1 = pop_basis[, 1], pop_ns2 = pop_basis[, 2], pop_ns3 = pop_basis[, 3],
    gdp_within = scaled$gdp_within,
    pm25_within_aug = scaled$pm25_within_aug,
    urban_share_within = scaled$urban_share_within,
    built_fraction_within = scaled$built_fraction_within,
    travel_time_within = scaled$travel_time_within,
    temperature_c_within_aug = scaled$temperature_c_within_aug,
    relative_humidity_pct_within_aug = scaled$relative_humidity_pct_within_aug,
    elevation_m_within_aug = scaled$elevation_m_within_aug,
    age20_39_share_within_aug = scaled$age20_39_share_within_aug,
    age60plus_share_within_aug = scaled$age60plus_share_within_aug,
    female_adult_share_within_aug = scaled$female_adult_share_within_aug,
    ntl_by_female = scaled$ntl_within * female,
    pop_by_female = scaled$pop_density_within * female,
    ntl_by_age = scaled$ntl_within * age_scaled,
    pop_by_age = scaled$pop_density_within * age_scaled
  )
  if (!identical(colnames(X), bundle$feature_names)) stop("Feature order mismatch")
  if (any(!is.finite(X))) stop("Non-finite future design matrix")
  list(
    X = X,
    ntl_z = scaled$ntl_within,
    pop_z = scaled$pop_density_within,
    country_id = match(data$iso3, recipe$country_levels),
    stage_id = match(data$development_stage, recipe$stage_levels)
  )
}

predict_uncalibrated <- function(data, target) {
  design <- make_design(data)
  offset <- matrix(
    rep(log(target[nonreference_labels] / target[["bmi_20_25"]]), each = nrow(data)),
    nrow = nrow(data), byrow = FALSE
  )
  eta <- offset + design$X %*% beta
  country_seen <- !is.na(design$country_id)
  stage_seen <- !is.na(design$stage_id)
  for (q in seq_along(nonreference_labels)) {
    if (any(country_seen)) {
      rows <- which(country_seen)
      ids <- design$country_id[rows]
      eta[rows, q] <- eta[rows, q] +
        sigma_country_intercept[q] * z_country_intercept[q, ids] +
        design$ntl_z[rows] * sigma_country_ntl[q] * z_country_ntl[q, ids] +
        design$pop_z[rows] * sigma_country_pop[q] * z_country_pop[q, ids]
    }
    if (any(stage_seen)) {
      rows <- which(stage_seen)
      ids <- design$stage_id[rows]
      eta[rows, q] <- eta[rows, q] +
        design$ntl_z[rows] * sigma_stage_ntl[q] * z_stage_ntl[q, ids] +
        design$pop_z[rows] * sigma_stage_pop[q] * z_stage_pop[q, ids]
    }
  }
  eta_full <- matrix(0, nrow(data), length(bmi_labels))
  eta_full[, bmi_labels != "bmi_20_25"] <- eta
  maximum <- apply(eta_full, 1L, max)
  probability <- exp(eta_full - maximum)
  probability / rowSums(probability)
}

training <- fread(training_path, select = scaled_columns)
training_ranges <- rbindlist(lapply(scaled_columns, function(column) {
  data.table(
    variable = column,
    train_min = min(training[[column]], na.rm = TRUE),
    train_max = max(training[[column]], na.rm = TRUE)
  )
}))
rm(training)

prior_root <- file.path(gate_c, "data", "0p5deg", "worldpop_2025_prior")
prior_values <- list()
for (sex_code in c("f", "m")) {
  for (age in seq(25L, 90L, 5L)) {
    path <- file.path(prior_root, sprintf("global_%s_%d_2025_0p5deg.tif", sex_code, age))
    if (!file.exists(path)) stop("Missing WorldPop prior: ", path)
    prior_values[[paste(sex_code, age, sep = "_")]] <- values(rast(path), mat = FALSE)
  }
}
prior_adult_total <- Reduce(`+`, lapply(prior_values, function(value) {
  value[!is.finite(value) | value < 0] <- 0
  value
}))

elevation_all <- values(rast(file.path(
  gate_c, "data", "0p5deg", "static", "elevation_m_0p5deg.tif"
)), mat = FALSE)
travel_all <- values(rast(file.path(
  gate_c, "data", "0p5deg", "static", "travel_time_city50k_minutes_0p5deg.tif"
)), mat = FALSE)

load_raster_values <- function(path) {
  if (!file.exists(path)) stop("Missing dynamic raster: ", path)
  values(rast(path), mat = FALSE)
}

load_dynamic_raw <- function(scenario, year) {
  ssp <- unname(scenarios[[scenario]])
  paths <- c(
    population = file.path(
      gate_c, "data", "0p5deg", "population",
      sprintf("population_%s_%d_0p5deg.tif", ssp, year)
    ),
    gdp = file.path(
      gdp_remediation, "data", "0p5deg",
      sprintf("gdp_spatial_complete_%s_%d_0p5deg.tif", ssp, year)
    ),
    ntl = file.path(
      ntl_root, "data", "0p5deg",
      sprintf("ntl_log_bridge1p5_%s_%d_0p5deg.tif", ssp, year)
    ),
    pm25 = file.path(
      full_covariates, "data", "0p5deg", "pm25_mmr_trailing5y_v2", "future",
      sprintf("pm25_mmr_ugkg_trailing5y_%s_%d_0p5deg.tif", ssp, year)
    ),
    urban = file.path(
      full_covariates, "data", "0p5deg", "urban_share",
      sprintf("urban_share_%s_%d_0p5deg.tif", ssp, year)
    ),
    built = file.path(
      full_covariates, "data", "0p5deg", "built_fraction_proxy",
      sprintf("built_fraction_proxy_%s_%d_0p5deg.tif", ssp, year)
    ),
    temperature = file.path(
      gate_c, "data", "0p5deg", "climate",
      sprintf("tas_c_%s_%d_0p5deg.tif", ssp, year)
    ),
    humidity = file.path(
      gate_c, "data", "0p5deg", "climate",
      sprintf("hurs_pct_%s_%d_0p5deg.tif", ssp, year)
    )
  )
  output <- lapply(paths, load_raster_values)
  output$paths <- paths
  output
}

dynamic_baseline_2025 <- lapply(names(scenarios), function(scenario) {
  load_dynamic_raw(scenario, 2025L)
})
names(dynamic_baseline_2025) <- names(scenarios)

load_dynamic <- function(scenario, year) {
  current <- load_dynamic_raw(scenario, year)
  scenario_base <- dynamic_baseline_2025[[scenario]]
  common_base <- dynamic_baseline_2025[["SSP2-4.5"]]
  if (scenario == "SSP2-4.5") return(current)
  if (year == 2025L) {
    common_start <- common_base
    common_start$paths <- current$paths
    return(common_start)
  }

  # Count-like fields are bridged on log1p scale; continuous fields use the
  # scenario change from their own 2025 node added to the common 2025 base.
  for (variable in c("population", "gdp")) {
    current[[variable]] <- expm1(
      log1p(pmax(common_base[[variable]], 0)) +
        log1p(pmax(current[[variable]], 0)) -
        log1p(pmax(scenario_base[[variable]], 0))
    )
  }
  for (variable in c("ntl", "pm25", "urban", "built", "temperature", "humidity")) {
    current[[variable]] <- common_base[[variable]] +
      current[[variable]] - scenario_base[[variable]]
  }
  current$population <- pmax(current$population, 0)
  current$gdp <- pmax(current$gdp, 0)
  current$ntl <- pmax(current$ntl, 0)
  current$pm25 <- pmax(current$pm25, 0)
  current$urban <- pmin(1, pmax(0, current$urban))
  current$built <- pmin(1, pmax(0, current$built))
  current$humidity <- pmin(100, pmax(0, current$humidity))
  current
}

build_spatial_input <- function(iso3, scenario, year, dynamic) {
  iso3_value <- iso3
  scenario_value <- scenario
  year_value <- year
  country_id <- country_lookup[iso3 == iso3_value, country_id]
  if (length(country_id) != 1L) stop("Missing country grid identifier: ", iso3)
  country_cells <- which(country_values == country_id)
  if (!length(country_cells)) stop("No represented cells: ", iso3)

  dynamic_population <- pmax(dynamic$population[country_cells], 0)
  dynamic_population[!is.finite(dynamic_population)] <- 0
  baseline_total <- prior_adult_total[country_cells]
  allocation_pre <- matrix(
    0, nrow = length(country_cells), ncol = length(age_groups) * length(sexes)
  )
  allocation_names <- character(ncol(allocation_pre))
  column_index <- 0L
  for (sex in sexes) {
    sex_code <- if (sex == "Female") "f" else "m"
    for (age_group in age_groups) {
      sex_value <- sex
      age_group_value <- age_group
      column_index <- column_index + 1L
      lower <- min(age_lower(age_group), 90L)
      group_prior <- prior_values[[paste(sex_code, lower, sep = "_")]][country_cells]
      group_prior[!is.finite(group_prior) | group_prior < 0] <- 0
      age_sex_share <- numeric(length(country_cells))
      have_profile <- is.finite(baseline_total) & baseline_total > 0
      age_sex_share[have_profile] <- group_prior[have_profile] / baseline_total[have_profile]
      raw_weight <- dynamic_population * age_sex_share
      if (sum(raw_weight) <= 0) raw_weight <- group_prior
      if (sum(raw_weight) <= 0) raw_weight <- dynamic_population
      if (sum(raw_weight) <= 0) raw_weight <- rep(1, length(country_cells))
      total <- anchor[
        J(scenario_value, iso3_value, year_value, sex_value, age_group_value),
        population_persons
      ]
      if (length(total) != 1L || !is.finite(total) || total < 0) {
        stop("Missing national population: ", scenario, " ", iso3, " ", year, " ", sex, " ", age_group)
      }
      if (total > 0) allocation_pre[, column_index] <- total * raw_weight / sum(raw_weight)
      allocation_names[column_index] <- paste(sex, age_group, sep = "|")
    }
  }
  colnames(allocation_pre) <- allocation_names
  adult_pre <- rowSums(allocation_pre)
  density_pre <- adult_pre / cell_area[country_cells]
  keep <- is.finite(density_pre) & density_pre >= 0.1 & adult_pre > 0
  if (!any(keep)) stop("No cells survive population support: ", scenario, " ", iso3, " ", year)
  retained_fraction <- sum(adult_pre[keep]) / sum(adult_pre)

  cells <- country_cells[keep]
  allocation <- allocation_pre[keep, , drop = FALSE]
  for (column_index in seq_len(ncol(allocation))) {
    total <- sum(allocation_pre[, column_index])
    if (total > 0) allocation[, column_index] <-
      total * allocation[, column_index] / sum(allocation[, column_index])
  }
  adult_population <- rowSums(allocation)

  age_lowers <- rep(vapply(age_groups, age_lower, integer(1)), times = length(sexes))
  sex_vector <- rep(sexes, each = length(age_groups))
  age20_39_share <- rowSums(allocation[, age_lowers < 40L, drop = FALSE]) / adult_population
  age60plus_share <- rowSums(allocation[, age_lowers >= 60L, drop = FALSE]) / adult_population
  female_share <- rowSums(allocation[, sex_vector == "Female", drop = FALSE]) / adult_population

  population_cell <- dynamic$population[cells]
  population_fallback <- !is.finite(population_cell) | population_cell <= 0
  population_cell[population_fallback] <- adult_population[population_fallback]
  population_cell <- pmax(population_cell, 0)
  gdp_cell <- dynamic$gdp[cells]
  gdp_cell[!is.finite(gdp_cell) | gdp_cell < 0] <- 0
  raw_values <- list(
    ntl_within = dynamic$ntl[cells],
    pop_density_within = log1p(population_cell / cell_area[cells]),
    gdp_within = log1p(ifelse(population_cell > 0, gdp_cell / population_cell, NA_real_)),
    pm25_within_aug = dynamic$pm25[cells],
    urban_share_within = pmin(1, pmax(0, dynamic$urban[cells])),
    built_fraction_within = pmin(1, pmax(0, dynamic$built[cells])),
    travel_time_within = log1p(pmax(0, travel_all[cells])),
    temperature_c_within_aug = dynamic$temperature[cells],
    relative_humidity_pct_within_aug = pmin(100, pmax(0, dynamic$humidity[cells])),
    elevation_m_within_aug = elevation_all[cells],
    age20_39_share_within_aug = age20_39_share,
    age60plus_share_within_aug = age60plus_share,
    female_adult_share_within_aug = female_share
  )
  centered <- lapply(names(raw_values), function(variable) {
    weighted_center(
      raw_values[[variable]], adult_population, variable, iso3, scenario, year
    )
  })
  names(centered) <- names(raw_values)
  stage <- disposition[iso3 == iso3_value, development_stage_2023][1]
  if (is.na(stage) || !nzchar(stage)) stage <- NA_character_
  model_grid <- data.table(
    cell_id = cells,
    longitude = cell_xy[cells, 1],
    latitude = cell_xy[cells, 2],
    scenario = scenario,
    iso3 = iso3,
    year = year,
    development_stage = stage,
    cell_adult_population = adult_population,
    cell_total_population = population_cell
  )
  for (variable in scaled_columns) model_grid[, (variable) := centered[[variable]]$within]

  extrapolation <- rbindlist(lapply(scaled_columns, function(variable_name) {
    range_row <- training_ranges[variable == variable_name]
    outside <- model_grid[[variable_name]] < range_row$train_min |
      model_grid[[variable_name]] > range_row$train_max
    data.table(
      scenario = scenario, iso3 = iso3, year = year, variable = variable_name,
      future_min = min(model_grid[[variable_name]], na.rm = TRUE),
      future_max = max(model_grid[[variable_name]], na.rm = TRUE),
      cells_outside = sum(outside),
      adult_population_fraction_outside = weighted_mean_safe(
        as.numeric(outside), adult_population
      )
    )
  }))

  list(
    model_grid = model_grid,
    allocation = allocation,
    adult_population = adult_population,
    retained_fraction = retained_fraction,
    population_fallback_cells = sum(population_fallback),
    extrapolation = extrapolation,
    maximum_centering_error = max(vapply(
      centered, function(value) value$centering_error, numeric(1)
    )),
    imputed_covariate_values = sum(vapply(
      centered, function(value) value$imputed, numeric(1)
    ))
  )
}

predict_country_scenario_year <- function(iso3, scenario, year, dynamic) {
  iso3_value <- iso3
  scenario_value <- scenario
  year_value <- year
  spatial <- build_spatial_input(iso3, scenario, year, dynamic)
  model_grid <- spatial$model_grid
  allocation <- spatial$allocation
  adult_population <- spatial$adult_population
  overall_numerator <- matrix(0, nrow(model_grid), length(bmi_labels))
  maximum_anchor_error <- 0
  maximum_raking_iterations <- 0L
  maximum_row_closure_error <- 0

  for (sex in sexes) {
    for (age_group in age_groups) {
      sex_value <- sex
      age_group_value <- age_group
      demographic_population <- allocation[, paste(sex, age_group, sep = "|")]
      target_row <- anchor[
        J(scenario_value, iso3_value, year_value, sex_value, age_group_value)
      ]
      if (nrow(target_row) != 1L) {
        stop("Missing BMI anchor: ", scenario, " ", iso3, " ", year, " ", sex, " ", age_group)
      }
      target <- setNames(
        as.numeric(unlist(target_row[, ..target_columns], use.names = FALSE)),
        bmi_labels
      )
      prediction_data <- copy(model_grid)
      prediction_data[, `:=`(
        sex = sex,
        age_group = age_group,
        model_age_midpoint = model_age_midpoint(age_group)
      )]
      uncalibrated <- predict_uncalibrated(prediction_data, target)
      calibrated <- rake_composition(uncalibrated, target, demographic_population)
      probability <- calibrated$probability
      national_calibrated <- colSums(
        probability * demographic_population / sum(demographic_population)
      )
      maximum_anchor_error <- max(maximum_anchor_error, max(abs(national_calibrated - target)))
      maximum_raking_iterations <- max(maximum_raking_iterations, calibrated$iterations)
      maximum_row_closure_error <- max(
        maximum_row_closure_error, max(abs(rowSums(probability) - 1))
      )
      overall_numerator <- overall_numerator + probability * demographic_population
    }
  }

  overall_probability <- overall_numerator / adult_population
  output <- model_grid[, .(
    scenario, iso3, year, cell_id, longitude, latitude,
    development_stage, cell_adult_population, cell_total_population
  )]
  for (index in seq_along(bmi_labels)) {
    output[, paste0("predicted_", bmi_labels[index]) := overall_probability[, index]]
  }
  output[, predicted_obesity := rowSums(overall_probability[, 5:7, drop = FALSE])]
  output[, predicted_mean_bmi := as.numeric(overall_probability %*% support)]

  qa <- data.table(
    scenario = scenario, iso3 = iso3, year = year,
    supported_cells = nrow(output),
    allocated_adult_population = sum(adult_population),
    preliminary_population_retained_fraction = spatial$retained_fraction,
    maximum_anchor_error = maximum_anchor_error,
    maximum_raking_iterations = maximum_raking_iterations,
    maximum_probability_closure_error = max(
      maximum_row_closure_error, max(abs(rowSums(overall_probability) - 1))
    ),
    maximum_centering_error = spatial$maximum_centering_error,
    imputed_covariate_values = spatial$imputed_covariate_values,
    population_fallback_cells = spatial$population_fallback_cells,
    country_effect_applied = iso3 %in% recipe$country_levels,
    stage_effect_applied = !is.na(model_grid$development_stage[1]) &&
      model_grid$development_stage[1] %in% recipe$stage_levels
  )
  list(prediction = output, qa = qa, extrapolation = spatial$extrapolation)
}

if (Sys.getenv("BMI_ANNUAL_CORE_ONLY", unset = "0") != "1") {
started_at <- timestamp_utc()
prediction_parts <- list()
qa_parts <- list()
extrapolation_parts <- list()
part_index <- 0L
total_tasks <- length(scenarios) * length(pilot_years) * length(pilot_countries)

for (scenario in names(scenarios)) {
  for (year in pilot_years) {
    message("Loading ", scenario, " dynamic inputs for ", year)
    dynamic <- load_dynamic(scenario, year)
    for (iso3 in pilot_countries) {
      part_index <- part_index + 1L
      message(sprintf("[%d/%d] %s %s %d", part_index, total_tasks, scenario, iso3, year))
      result <- predict_country_scenario_year(iso3, scenario, year, dynamic)
      prediction_parts[[part_index]] <- result$prediction
      qa_parts[[part_index]] <- result$qa
      extrapolation_parts[[part_index]] <- result$extrapolation
    }
    rm(dynamic)
    gc()
  }
}

predictions <- rbindlist(prediction_parts)
qa <- rbindlist(qa_parts)
extrapolation <- rbindlist(extrapolation_parts)
setkey(predictions, scenario, iso3, year, cell_id)

prediction_path <- file.path(
  root, "predictions", "global_key_year_four_scenario_overall_0p5deg_pilot_v1.csv.gz"
)
qa_path <- file.path(root, "qa", "global_key_year_pilot_task_QA_v1.csv")
extrapolation_path <- file.path(root, "qa", "global_key_year_pilot_extrapolation_QA_v1.csv")
write_new_csv(predictions, prediction_path)
write_new_csv(qa, qa_path, compress = FALSE)
write_new_csv(extrapolation, extrapolation_path, compress = FALSE)

comparison_parts <- list()
comparison_index <- 0L
for (scenario in setdiff(names(scenarios), "SSP2-4.5")) {
  scenario_value <- scenario
  reference <- predictions[scenario == "SSP2-4.5"]
  alternative <- predictions[scenario == scenario_value]
  pair <- merge(
    reference, alternative,
    by = c("iso3", "year", "cell_id", "longitude", "latitude"),
    suffixes = c("_ssp2", "_alternative")
  )
  reference_probability <- as.matrix(
    pair[, paste0("predicted_", bmi_labels, "_ssp2"), with = FALSE]
  )
  alternative_probability <- as.matrix(
    pair[, paste0("predicted_", bmi_labels, "_alternative"), with = FALSE]
  )
  comparison_index <- comparison_index + 1L
  comparison_parts[[comparison_index]] <- data.table(
    scenario = scenario,
    iso3 = pair$iso3,
    year = pair$year,
    cell_id = pair$cell_id,
    longitude = pair$longitude,
    latitude = pair$latitude,
    common_population_weight = pair$cell_adult_population_ssp2,
    alternative_population_weight = pair$cell_adult_population_alternative,
    w2_bmi_kg_m2 = row_wasserstein2(reference_probability, alternative_probability),
    obesity_difference_pp = 100 * (
      pair$predicted_obesity_alternative - pair$predicted_obesity_ssp2
    ),
    absolute_obesity_difference_pp = 100 * abs(
      pair$predicted_obesity_alternative - pair$predicted_obesity_ssp2
    ),
    mean_bmi_difference = pair$predicted_mean_bmi_alternative -
      pair$predicted_mean_bmi_ssp2,
    seven_category_mae_pp = 100 * rowMeans(abs(
      alternative_probability - reference_probability
    ))
  )
}
comparison <- rbindlist(comparison_parts)
comparison_path <- file.path(root, "data", "global_key_year_cell_comparison_vs_ssp2_v1.csv.gz")
write_new_csv(comparison, comparison_path)

scenario_summary <- comparison[, .(
  matched_cells = .N,
  common_population = sum(common_population_weight),
  population_weighted_mean_w2_kg_m2 = weighted_mean_safe(
    w2_bmi_kg_m2, common_population_weight
  ),
  population_weighted_mean_obesity_difference_pp = weighted_mean_safe(
    obesity_difference_pp, common_population_weight
  ),
  population_weighted_mean_absolute_obesity_difference_pp = weighted_mean_safe(
    absolute_obesity_difference_pp, common_population_weight
  ),
  population_weighted_p99_absolute_obesity_difference_pp = weighted_quantile(
    absolute_obesity_difference_pp, common_population_weight, 0.99
  ),
  population_weighted_mean_bmi_difference = weighted_mean_safe(
    mean_bmi_difference, common_population_weight
  ),
  population_weighted_seven_category_mae_pp = weighted_mean_safe(
    seven_category_mae_pp, common_population_weight
  )
), by = .(scenario, year)]
scenario_summary_path <- file.path(root, "qa", "global_key_year_comparison_summary_v1.csv")
write_new_csv(scenario_summary, scenario_summary_path, compress = FALSE)

aggregate_parts <- list()
aggregate_index <- 0L
for (scenario in names(scenarios)) {
  scenario_value <- scenario
  for (year in pilot_years) {
    year_value <- year
    value <- predictions[scenario == scenario_value & year == year_value]
    probability <- as.matrix(value[, paste0("predicted_", bmi_labels), with = FALSE])
    weight <- value$cell_adult_population
    composition <- colSums(probability * weight) / sum(weight)
    aggregate_index <- aggregate_index + 1L
    aggregate_parts[[aggregate_index]] <- data.table(
      scenario = scenario,
      year = year,
      population_basis = "scenario_specific_pilot_population",
      adult_population = sum(weight),
      p_bmi_lt_18_5 = composition[1],
      p_bmi_18_5_20 = composition[2],
      p_bmi_20_25 = composition[3],
      p_bmi_25_30 = composition[4],
      p_bmi_30_35 = composition[5],
      p_bmi_35_40 = composition[6],
      p_bmi_ge_40 = composition[7],
      p_obesity_ge30 = sum(composition[5:7]),
      mean_bmi = sum(composition * support)
    )
  }
}
aggregate <- rbindlist(aggregate_parts)
aggregate_path <- file.path(root, "data", "global_key_year_aggregate_v1.csv")
write_new_csv(aggregate, aggregate_path, compress = FALSE)

maximum_anchor_error_pp <- 100 * max(qa$maximum_anchor_error)
maximum_closure_error <- max(qa$maximum_probability_closure_error)
maximum_centering_error <- max(qa$maximum_centering_error)
minimum_population_retained <- min(qa$preliminary_population_retained_fraction)
summary_2025 <- scenario_summary[year == 2025]

gate <- data.table(
  check_id = c(
    "P01_TASK_COMPLETENESS", "P02_NATIONAL_ANCHOR", "P03_COMPOSITION_CLOSURE",
    "P04_WITHIN_COUNTRY_CENTERING", "P05_POPULATION_MASK_RETENTION",
    "P06_COMMON_2025_W2", "P07_COMMON_2025_OBESITY"
  ),
  observed = c(
    nrow(qa), maximum_anchor_error_pp, maximum_closure_error,
    maximum_centering_error, minimum_population_retained,
    max(summary_2025$population_weighted_mean_w2_kg_m2),
    max(summary_2025$population_weighted_mean_absolute_obesity_difference_pp)
  ),
  threshold = c(
    total_tasks, limits$maximum_national_anchor_error_pp,
    limits$maximum_probability_closure_error, limits$maximum_centering_error,
    limits$minimum_country_scenario_year_population_retained,
    limits$maximum_2025_population_weighted_mean_w2_kg_m2,
    limits$maximum_2025_population_weighted_mean_absolute_obesity_difference_pp
  ),
  direction = c("equal", rep("less_or_equal", 3), "greater_or_equal", rep("less_or_equal", 2))
)
gate[, status := fifelse(
  (direction == "equal" & observed == threshold) |
    (direction == "less_or_equal" & observed <= threshold) |
    (direction == "greater_or_equal" & observed >= threshold),
  "PASS", "FAIL"
)]
gate_status <- if (all(gate$status == "PASS")) "PASS_WITH_WARN" else "FAIL"
gate_path <- file.path(root, "qa", "global_key_year_pilot_gate_v1.csv")
write_new_csv(gate, gate_path, compress = FALSE)

extrapolation_with_weight <- qa[
  extrapolation,
  on = .(scenario, iso3, year),
  .(
    scenario = i.scenario, iso3 = i.iso3, year = i.year,
    variable = i.variable, future_min = i.future_min, future_max = i.future_max,
    cells_outside = i.cells_outside,
    adult_population_fraction_outside = i.adult_population_fraction_outside,
    allocated_adult_population
  )
]
extrapolation_summary <- extrapolation_with_weight[, .(
  maximum_adult_population_fraction_outside = max(
    adult_population_fraction_outside, na.rm = TRUE
  ),
  population_weighted_mean_fraction_outside = weighted.mean(
    adult_population_fraction_outside, allocated_adult_population
  )
), by = .(scenario, year, variable)]
extrapolation_summary_path <- file.path(
  root, "qa", "global_key_year_pilot_extrapolation_summary_v1.csv"
)
write_new_csv(extrapolation_summary, extrapolation_summary_path, compress = FALSE)

report <- c(
  "# Full-covariate global key-year pilot v1",
  "",
  paste0("Status: **", gate_status, "**"),
  "",
  "## Scope",
  "",
  paste0(
    "The validated 22-feature Bayesian multinomial logistic-normal MRP was applied ",
    "to SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5 in 2025 and 2050 for ",
    length(pilot_countries), " spatially represented pilot countries."
  ),
  "National age-sex BMI anchors and national age-sex population totals are scenario-specific. Within-country allocation uses all eight future dynamic covariates.",
  "",
  "## Population support",
  "",
  "WorldPop 2025 age-sex profiles were redistributed using each scenario-year population surface. Cells below 0.1 adults per km2 were removed, after which each national age-sex population was recalibrated exactly over retained cells.",
  paste0("Minimum preliminary country-scenario-year population retention: ", sprintf("%.6f", minimum_population_retained), "."),
  "",
  "## QA",
  "",
  paste(capture.output(print(gate)), collapse = "\n"),
  "",
  paste0("Maximum national anchor error: ", format(maximum_anchor_error_pp, scientific = TRUE), " percentage points."),
  paste0("Maximum probability closure error: ", format(maximum_closure_error, scientific = TRUE), "."),
  paste0("Maximum within-country centering error: ", format(maximum_centering_error, scientific = TRUE), "."),
  "",
  "## Scenario separation",
  "",
  paste(capture.output(print(scenario_summary)), collapse = "\n"),
  "",
  "Scenario-specific population aggregates are reported separately from common-SSP2-population cell comparisons. The former describe total population-weighted burden-relevant change; the latter isolate changes in predicted BMI surfaces from population redistribution.",
  "",
  "## Authorization",
  "",
  if (gate_status == "PASS_WITH_WARN") {
    "The pilot authorizes preparation of the annual 2025-2050 0.5-degree production contract. Future covariate extrapolation remains a reportable warning."
  } else {
    "Annual production is not authorized until failed pilot checks are remediated and rerun."
  },
  "Disease-burden analysis remains unauthorized."
)
report_path <- file.path(root, "reports", "global_key_year_pilot_report_v1.md")
write_new_text(report, report_path)

summary_json <- list(
  status = gate_status,
  started_at_utc = started_at,
  completed_at_utc = timestamp_utc(),
  countries = length(pilot_countries),
  scenarios = names(scenarios),
  years = pilot_years,
  tasks = nrow(qa),
  prediction_rows = nrow(predictions),
  maximum_anchor_error_pp = maximum_anchor_error_pp,
  maximum_probability_closure_error = maximum_closure_error,
  minimum_population_retained = minimum_population_retained,
  annual_0p5deg_production_authorized = gate_status == "PASS_WITH_WARN",
  disease_burden_analysis_authorized = FALSE
)
summary_json_path <- file.path(root, "qa", "global_key_year_pilot_summary_v1.json")
write_new_json(summary_json, summary_json_path)

output_paths <- c(
  contract_path, prediction_path, qa_path, extrapolation_path, comparison_path,
  scenario_summary_path, aggregate_path, gate_path, extrapolation_summary_path,
  report_path, summary_json_path,
  file.path(root, "scripts", "run_full_covariate_global_key_year_pilot_v1.R")
)
checksum <- data.table(
  path = normalizePath(output_paths, winslash = "/", mustWork = TRUE),
  bytes = file.info(output_paths)$size,
  sha256 = vapply(output_paths, sha256_file, character(1))
)
checksum_path <- file.path(root, "checksums", "global_key_year_pilot_SHA256_v1.csv")
write_new_csv(checksum, checksum_path, compress = FALSE)

cat(toJSON(summary_json, pretty = TRUE, auto_unbox = TRUE), "\n")
}
