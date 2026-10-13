#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(exactextractr)
  library(sf)
  library(terra)
})

options(stringsAsFactors = FALSE, warn = 1)
terraOptions(progress = 1, memfrac = 0.65)

future_root <- file.path(.cfg$ROOT, ".codex_tmp", "future_results_v1")
historic_root <- file.path(.cfg$ROOT, ".codex_tmp", "historic_results_v1")
ascii_inputs <- file.path(.cfg$ROOT, ".codex_tmp", "full_mrp_refit_inputs_v1")

run_dir <- file.path(
  future_root, "04_future_capable_mrp",
  "full_covariate_mrp_refit_v1_20260905"
)
full_covariate_dir <- file.path(
  future_root, "02_harmonized_covariates",
  "full_covariate_extension_v1_20260905"
)
gate_c <- file.path(
  future_root, "02_harmonized_covariates",
  "gate_c_v1_country_coverage_remediated_20260828_122619"
)
training_path <- file.path(
  historic_root, "experiments", "BMI7_MRP_STEPS_regional_augmentation_20260812",
  "data", "bmi7_enhanced_plus_steps_regional_training_v1.csv.gz"
)
age_pop_dir <- file.path(historic_root, "cache", "auxiliary_mrp_0p5deg", "age_sex_population")
pm_dir <- file.path(
  full_covariate_dir, "data", "0p5deg", "pm25_mmr_trailing5y_v2",
  "historical_common"
)

dirs <- list(
  data = file.path(run_dir, "data"),
  qa = file.path(run_dir, "qa"),
  logs = file.path(run_dir, "logs"),
  contracts = file.path(run_dir, "contracts")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

output_training <- file.path(dirs$data, "bmi7_full_covariate_training_v1.csv.gz")
if (file.exists(output_training)) stop("Refusing to overwrite: ", output_training)

log_path <- file.path(dirs$logs, "rebuild_training_pm25_mmr_v1.log")
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

required <- c(
  training_path,
  file.path(ascii_inputs, "new_bmi_source_regions.gpkg"),
  file.path(ascii_inputs, "steps_regional_training_polygons_v1.gpkg"),
  file.path(gate_c, "data", "0p5deg", "country_id_0p5deg.tif"),
  file.path(gate_c, "data", "0p5deg", "country_id_lookup.csv")
)
if (any(!file.exists(required))) stop("Missing required input: ", paste(required[!file.exists(required)], collapse = "; "))

stamp("Reading frozen enhanced training table")
training <- fread(training_path, encoding = "UTF-8")
training[, `:=`(
  source = as.character(source),
  source_family = as.character(source_family),
  iso3 = toupper(as.character(iso3)),
  geography_id = as.character(geography_id),
  year = as.integer(year)
)]
training[, pm_geography_id := geography_id]
training[source == "Mexico_ENSANUT_state", pm_geography_id := sprintf("%02d", as.integer(geography_id))]

years <- sort(unique(training$year))
if (min(years) < 2006L || max(years) > 2024L) stop("Unexpected training-year range")

country_id <- rast(file.path(gate_c, "data", "0p5deg", "country_id_0p5deg.tif"))
country_lookup <- fread(file.path(gate_c, "data", "0p5deg", "country_id_lookup.csv"))
country_lookup[, `:=`(country_id = as.integer(country_id), iso3 = toupper(as.character(iso3)))]

standard_polygons <- st_read(
  file.path(ascii_inputs, "new_bmi_source_regions.gpkg"),
  layer = "regions", quiet = TRUE
)
standard_polygons$source <- as.character(standard_polygons$source)
standard_polygons$iso3 <- toupper(as.character(standard_polygons$iso3))
standard_polygons$geography_id <- as.character(standard_polygons$geography_id)
standard_polygons$pm_geography_id <- standard_polygons$geography_id
standard_polygons <- st_transform(st_make_valid(standard_polygons), 4326)

steps_polygons <- st_read(
  file.path(ascii_inputs, "steps_regional_training_polygons_v1.gpkg"),
  quiet = TRUE
)
steps_polygons$source <- "WHO_STEPS_region_measured_BMI"
steps_polygons$iso3 <- toupper(as.character(steps_polygons$iso3))
steps_polygons$geography_id <- as.character(steps_polygons$region_id)
steps_polygons$pm_geography_id <- steps_polygons$geography_id
steps_polygons <- st_transform(st_make_valid(steps_polygons), 4326)

geo <- unique(training[, .(
  source, source_family, iso3, geography_id, year,
  pm_geography_id,
  grid_lon = as.numeric(grid_lon), grid_lat = as.numeric(grid_lat)
)])

adult_population <- function(year) {
  population_year <- max(2015L, as.integer(year))
  files <- list.files(
    age_pop_dir,
    pattern = sprintf("_%d_0p5deg[.]tif$", population_year),
    full.names = TRUE
  )
  if (length(files) != 28L) {
    stop("Expected 28 age-sex population rasters for ", population_year, "; found ", length(files))
  }
  pop <- sum(rast(files), na.rm = TRUE)
  pop[pop <= 0] <- NA
  if (!compareGeom(pop, country_id, stopOnError = FALSE)) {
    pop <- resample(pop, country_id, method = "bilinear")
  }
  names(pop) <- "adult_population"
  list(raster = pop, year = population_year)
}

country_means <- function(pm, pop, year, population_year) {
  s <- c(country_id, pm, pop)
  names(s) <- c("country_id", "pm", "pop")
  d <- as.data.table(values(s, dataframe = TRUE, na.rm = FALSE))
  d <- d[is.finite(country_id) & is.finite(pm) & is.finite(pop) & pop > 0]
  out <- d[, .(
    country_population_weight = sum(pop),
    pm25_mmr_country_ugkg = sum(pm * pop) / sum(pop),
    contributing_cells = .N
  ), by = country_id]
  out <- merge(out, country_lookup, by = "country_id", all.x = TRUE, sort = FALSE)
  out[, `:=`(year = as.integer(year), population_year_used = as.integer(population_year))]
  out[]
}

polygon_extract <- function(polygons, pm, pop) {
  if (!nrow(polygons)) return(data.table())
  den <- ifel(is.finite(pm) & is.finite(pop) & pop > 0, pop, NA)
  num <- pm * den
  stack <- c(den, num)
  names(stack) <- c("den", "num")
  values <- as.data.table(exact_extract(stack, polygons, fun = "sum", progress = FALSE))
  attrs <- as.data.table(st_drop_geometry(polygons))
  out <- cbind(
    attrs[, .(source, iso3, pm_geography_id)],
    values
  )
  out[, `:=`(
    pm25_mmr_geo_ugkg = `sum.num` / `sum.den`,
    extraction_population = `sum.den`,
    extraction_method = "polygon_adult_population_weighted"
  )]
  out[, .(source, iso3, pm_geography_id, pm25_mmr_geo_ugkg, extraction_population, extraction_method)]
}

geo_results <- list()
country_results <- list()
result_index <- 0L
country_index <- 0L

for (year in years) {
  current_year <- as.integer(year)
  stamp("Processing training year ", year)
  pm_path <- file.path(pm_dir, sprintf("pm25_mmr_ugkg_trailing5y_common_%d_0p5deg.tif", year))
  if (!file.exists(pm_path)) stop("Missing historical PM2.5 raster: ", pm_path)
  pm <- rast(pm_path)
  if (!compareGeom(pm, country_id, stopOnError = FALSE)) {
    pm <- resample(pm, country_id, method = "bilinear")
  }
  names(pm) <- "pm"
  pop_info <- adult_population(year)
  pop <- pop_info$raster

  cm <- country_means(pm, pop, year, pop_info$year)
  cm <- cm[iso3 %chin% unique(training[year == current_year, iso3])]
  country_index <- country_index + 1L
  country_results[[country_index]] <- cm

  gy <- geo[year == current_year]
  handled <- character()

  national <- gy[source_family == "WHO STEPS"]
  if (nrow(national)) {
    national <- merge(
      national,
      cm[, .(iso3, pm25_mmr_country_ugkg)],
      by = "iso3", all.x = TRUE, sort = FALSE
    )
    national[, `:=`(
      pm25_mmr_geo_ugkg = pm25_mmr_country_ugkg,
      extraction_population = NA_real_,
      extraction_method = "national_record_within_effect_zero"
    )]
    result_index <- result_index + 1L
    geo_results[[result_index]] <- national[, .(
      source, iso3, geography_id, year,
      pm25_mmr_geo_ugkg, extraction_population, extraction_method
    )]
    handled <- c(handled, unique(national$source))
  }

  points <- gy[source == "DHS_GPS"]
  if (nrow(points)) {
    if (any(!is.finite(points$grid_lon) | !is.finite(points$grid_lat))) stop("Missing DHS coordinates in ", year)
    point_values <- terra::extract(pm, as.matrix(points[, .(grid_lon, grid_lat)]), method = "bilinear")
    points[, `:=`(
      pm25_mmr_geo_ugkg = as.numeric(point_values[[ncol(point_values)]]),
      extraction_population = NA_real_,
      extraction_method = "point_bilinear"
    )]
    result_index <- result_index + 1L
    geo_results[[result_index]] <- points[, .(
      source, iso3, geography_id, year,
      pm25_mmr_geo_ugkg, extraction_population, extraction_method
    )]
    handled <- c(handled, "DHS_GPS")
  }

  standard_sources <- intersect(unique(gy$source), unique(standard_polygons$source))
  for (source_name in standard_sources) {
    expected <- gy[source == source_name, unique(pm_geography_id)]
    polygons <- standard_polygons[
      standard_polygons$source == source_name & standard_polygons$pm_geography_id %in% expected,
    ]
    missing <- setdiff(expected, polygons$pm_geography_id)
    if (length(missing)) stop("Missing standard polygons for ", source_name, ": ", paste(missing, collapse = ", "))
    out <- polygon_extract(polygons, pm, pop)
    out <- merge(
      gy[source == source_name, .(source, iso3, geography_id, pm_geography_id)],
      out,
      by = c("source", "iso3", "pm_geography_id"),
      all.x = TRUE, sort = FALSE
    )
    out[, pm_geography_id := NULL]
    out[, year := as.integer(year)]
    result_index <- result_index + 1L
    geo_results[[result_index]] <- out
    handled <- c(handled, source_name)
  }

  if ("WHO_STEPS_region_measured_BMI" %in% gy$source) {
    expected <- gy[source == "WHO_STEPS_region_measured_BMI", unique(pm_geography_id)]
    polygons <- steps_polygons[steps_polygons$pm_geography_id %in% expected, ]
    missing <- setdiff(expected, polygons$pm_geography_id)
    if (length(missing)) stop("Missing STEPS regional polygons: ", paste(missing, collapse = ", "))
    out <- polygon_extract(polygons, pm, pop)
    out <- merge(
      gy[source == "WHO_STEPS_region_measured_BMI", .(source, iso3, geography_id, pm_geography_id)],
      out,
      by = c("source", "iso3", "pm_geography_id"),
      all.x = TRUE, sort = FALSE
    )
    out[, pm_geography_id := NULL]
    out[, year := as.integer(year)]
    result_index <- result_index + 1L
    geo_results[[result_index]] <- out
    handled <- c(handled, "WHO_STEPS_region_measured_BMI")
  }

  unhandled <- setdiff(unique(gy$source), unique(handled))
  if (length(unhandled)) stop("Unhandled training source(s) in ", year, ": ", paste(unhandled, collapse = ", "))
  rm(pm, pop, pop_info, cm, gy); gc(verbose = FALSE)
}

geo_values <- rbindlist(geo_results, use.names = TRUE, fill = TRUE)
country_values <- rbindlist(country_results, use.names = TRUE, fill = TRUE)
if (anyDuplicated(geo_values, by = c("source", "iso3", "geography_id", "year"))) {
  stop("Duplicate reconstructed geography-year PM2.5 rows")
}

stamp("Joining reconstructed PM2.5 to all training records")
legacy <- training[, .(
  row_number = .I,
  pm25_concentration_aug_legacy = as.numeric(pm25_aug),
  pm25_concentration_within_aug_legacy = as.numeric(pm25_within_aug),
  pm25_concentration_model_feature_aug_legacy = as.numeric(pm25_model_feature_aug)
)]
training <- cbind(training, legacy[, -"row_number"])
training <- merge(
  training,
  geo_values,
  by = c("source", "iso3", "geography_id", "year"),
  all.x = TRUE, sort = FALSE
)
training[, pm_geography_id := NULL]
training <- merge(
  training,
  country_values[, .(iso3, year, pm25_mmr_country_ugkg, population_year_used)],
  by = c("iso3", "year"),
  all.x = TRUE, sort = FALSE
)

# National STEPS observations have no estimable within-country spatial offset.
# For very small island countries the 0.5-degree country mask has no finite
# population cell; keep the within-country feature explicitly at zero rather
# than inventing an absolute country PM2.5 value.
training[source_family == "WHO STEPS" & !is.finite(pm25_mmr_geo_ugkg), `:=`(
  pm25_mmr_geo_ugkg = 0,
  pm25_mmr_country_ugkg = 0,
  pm25_national_no_country_grid_flag = TRUE
)]
training[is.na(pm25_national_no_country_grid_flag), pm25_national_no_country_grid_flag := FALSE]

if (nrow(training) != nrow(legacy)) stop("Training row count changed during PM2.5 join")
bad_geo <- training[!is.finite(pm25_mmr_geo_ugkg), .(
  rows = .N,
  geography_ids = paste(head(unique(geography_id), 20L), collapse = "|"),
  years = paste(sort(unique(year)), collapse = "|"),
  sources = paste(unique(source), collapse = "|")
), by = .(source_family, iso3)]
bad_country <- training[!is.finite(pm25_mmr_country_ugkg), .(
  rows = .N,
  years = paste(sort(unique(year)), collapse = "|"),
  sources = paste(unique(source), collapse = "|")
), by = .(source_family, iso3)]
fwrite(bad_geo, file.path(dirs$qa, "PM25_training_nonfinite_geography_diagnostic_v1.csv"), bom = TRUE)
fwrite(bad_country, file.path(dirs$qa, "PM25_training_nonfinite_country_diagnostic_v1.csv"), bom = TRUE)
if (nrow(bad_geo)) stop("Non-finite reconstructed geography PM2.5 values; see diagnostic CSV")
if (nrow(bad_country)) stop("Non-finite reconstructed country PM2.5 values; see diagnostic CSV")

training[, pm25_mmr_within_aug := pm25_mmr_geo_ugkg - pm25_mmr_country_ugkg]
training[source_family == "WHO STEPS", pm25_mmr_within_aug := 0]
training[, `:=`(
  pm25_aug = pm25_mmr_geo_ugkg,
  pm25_within_aug = pm25_mmr_within_aug,
  pm25_model_feature_aug = pm25_mmr_within_aug,
  pm25_year_used_aug = year,
  pm25_proxy_flag_aug = population_year_used != year,
  pm25_source_aug = "NorESM2-LM near-surface PM2.5 mass mixing ratio; trailing-five-year mean"
)]

setorder(training, source_role, source_family, survey_id, geography_id, sex, age_midpoint)

qa_source <- training[, .(
  rows = .N,
  countries = uniqueN(iso3),
  geographies = uniqueN(geography_id),
  finite_new_pm = sum(is.finite(pm25_within_aug)),
  new_pm_mean = mean(pm25_within_aug),
  new_pm_sd = sd(pm25_within_aug),
  new_pm_min = min(pm25_within_aug),
  new_pm_max = max(pm25_within_aug),
  legacy_new_correlation = cor(pm25_concentration_within_aug_legacy, pm25_within_aug),
  population_proxy_rows = sum(pm25_proxy_flag_aug)
), by = source_family]
qa_source[, status := fifelse(finite_new_pm == rows, "PASS", "FAIL")]

qa_method <- unique(training[, .(
  source_family, source, extraction_method,
  population_year_used, year
)])[, .(
  geography_years = .N,
  training_year_min = min(year),
  training_year_max = max(year),
  population_year_min = min(population_year_used),
  population_year_max = max(population_year_used)
), by = .(source_family, source, extraction_method)]

national_max <- training[source_family == "WHO STEPS", max(abs(pm25_within_aug))]
overall_qa <- data.table(
  check = c(
    "training_row_count", "finite_pm25_feature", "national_steps_within_zero",
    "geography_year_unique", "year_coverage"
  ),
  observed = c(
    as.character(nrow(training)),
    as.character(sum(is.finite(training$pm25_within_aug))),
    format(national_max, scientific = TRUE),
    as.character(nrow(geo_values)),
    paste(range(training$year), collapse = "-")
  ),
  expected = c(
    "53777", "53777", "0", as.character(nrow(geo)), "2006-2024"
  )
)
overall_qa[, status := c(
  ifelse(nrow(training) == 53777L, "PASS", "FAIL"),
  ifelse(all(is.finite(training$pm25_within_aug)), "PASS", "FAIL"),
  ifelse(national_max < 1e-12, "PASS", "FAIL"),
  ifelse(nrow(geo_values) == nrow(geo), "PASS", "FAIL"),
  ifelse(min(training$year) == 2006L && max(training$year) == 2024L, "PASS", "FAIL")
)]

fwrite(geo_values, file.path(dirs$data, "pm25_mmr_training_geography_values_v1.csv"), bom = TRUE)
fwrite(country_values, file.path(dirs$data, "pm25_mmr_country_year_means_v1.csv"), bom = TRUE)
fwrite(qa_source, file.path(dirs$qa, "PM25_training_reconstruction_by_source_QA_v1.csv"), bom = TRUE)
fwrite(qa_method, file.path(dirs$qa, "PM25_training_reconstruction_methods_QA_v1.csv"), bom = TRUE)
fwrite(overall_qa, file.path(dirs$qa, "PM25_training_reconstruction_overall_QA_v1.csv"), bom = TRUE)

if (any(overall_qa$status == "FAIL") || any(qa_source$status == "FAIL")) {
  stop("Training PM2.5 reconstruction QA failed")
}

fwrite(training, output_training, compress = "gzip", bom = TRUE)
stamp("PASS: wrote ", nrow(training), " rows to ", output_training)
