if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

try(Sys.setlocale("LC_CTYPE", "Chinese"), silent = TRUE)
source(
  file.path(file.path(.cfg$ROOT, "scripts", "bmi_downscaling"), "00_config.R"),
  encoding = "UTF-8"
)

# R on this host cannot reliably retain Chinese path literals under the
# default locale.  Use ASCII aliases for existing directories only.
RESULT_ROOT <- file.path(.cfg$ROOT, "downscale_ascii")
PATHS$parameters <- file.path(RESULT_ROOT, "01_parameters")
PATHS$bmi_grids <- file.path(RESULT_ROOT, "05_bmi_grids")
PATHS$cache <- file.path(RESULT_ROOT, "cache")
PATHS$population_dir <- file.path(.cfg$DATA, "人口密度网格数据", "global_pop")
PATHS$historical_population_dir <- file.path(.cfg$DATA, "墨西哥_巴西WorldPoP")
PATHS$age_sex_population_dir <- file.path(.cfg$DATA, "人口结构网格数据", "downloads")
PATHS$smod_dir <- file.path(.cfg$DATA, "城市化网格数据_GHS_SMOD")
PATHS$builtup_dir <- file.path(.cfg$DATA, "建筑覆盖 GHS built-up surface")
PATHS$travel_time_dir <- file.path(.cfg$DATA, "Travel time to cities and ports in the year 2015")

suppressPackageStartupMessages({
  library(data.table)
  library(rstan)
  library(splines)
  library(terra)
})

args <- commandArgs(trailingOnly = TRUE)
year_arg <- sub("^--years=", "", args[grepl("^--years=", args)])
YEARS <- if (length(year_arg)) {
  as.integer(strsplit(year_arg[1], ",", fixed = TRUE)[[1]])
} else {
  2015:2024
}
resolution_arg <- sub(
  "^--resolution=",
  "",
  args[grepl("^--resolution=", args)]
)
GRID_RESOLUTION <- if (length(resolution_arg)) {
  as.numeric(resolution_arg[1])
} else {
  0.1
}
if (!identical(GRID_RESOLUTION, 0.1)) {
  stop("Stage 9 predicts only at 0.1 degrees; 0.5 degrees is aggregated later", call. = FALSE)
}
RESOLUTION_LABEL <- "0p1deg"
COUNTRY_GRID_MEMBER <- "source_1km"
overwrite <- "--overwrite" %in% args
test_mode <- "--test" %in% args
if (test_mode && !length(year_arg)) YEARS <- 2020L

seed <- 20260803L
set.seed(seed)

stage09_root <- file.path(
  file.path(.cfg$ROOT, "future_scenario_ascii"), "05_historical_bmi_1990_2024",
  "historical_25plus_compat_v1_20260909"
)
stage03_root <- file.path(
  RESULT_ROOT, "experiments", "BMI7_MRP_STEPS_regional_augmentation_20260812",
  "mrp_regional_validation_v1"
)
stage01_root <- file.path(
  RESULT_ROOT, "experiments", "BMI7_model_comparison_20260728",
  "gated_run_v1", "stage_01_data"
)
experiment_root <- file.path(
  RESULT_ROOT, "experiments", "BMI7_model_comparison_20260728"
)
model_recipe_path <- file.path(
  stage03_root, "models", "checkpoints", "full_model_recipe_v1.rds"
)
model_fit_path <- file.path(
  stage03_root, "models", "checkpoints", "full_model_fit_v1.rds"
)
ncd_path <- file.path(
  PATHS$parameters,
  "ncd_risc_2015_2024_bmi7_age_sex.csv"
)
context_path <- file.path(
  PATHS$parameters,
  "country_context_mrp_20260725.csv"
)
country_object_root <- file.path(PATHS$bmi_grids, "..", "02_derived")
country_object_root <- normalizePath(
  country_object_root,
  winslash = "/",
  mustWork = TRUE
)
output_root <- file.path(stage09_root, "data", RESOLUTION_LABEL)
derived_root <- file.path(stage09_root, "derived", RESOLUTION_LABEL)
cache_root <- file.path(stage09_root, "cache", RESOLUTION_LABEL)
upstream_cache_root <- file.path(PATHS$cache, "auxiliary_mrp_0p1deg")
climate_cache_root <- file.path(stage09_root, "input_cache", "climate_0p1deg")
age_cache_old_root <- file.path(
  file.path(.cfg$ROOT, "downscale_ascii", "cache", "historical_25plus_age_structure_0p1deg")
)
age_cache_new_root <- age_cache_old_root
elevation_path <- file.path(
  file.path(.cfg$ROOT, "future_scenario_ascii", "06_future_bmi_2025_2050"),
  "phase6g2_0p1deg_harmonization_pilot_v4_20260902",
  "data", "0p1deg", "static", "elevation_m_0p1deg_nested_v1.tif"
)
metadata_root <- file.path(output_root, "metadata")
for (directory in c(
  output_root, derived_root, cache_root, metadata_root,
  file.path(stage09_root, "cache", "terra")
)) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}

for (path in c(
  model_recipe_path, model_fit_path, ncd_path, context_path,
  elevation_path
)) {
  assert_file(path)
}

terraOptions(
  tempdir = file.path(stage09_root, "cache", "terra"),
  memfrac = 0.65,
  progress = if (test_mode) 1 else 0
)

model_categories <- c(
  "bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
  "bmi_30_35", "bmi_35_40", "bmi_ge_40"
)
output_categories <- c(
  "bmi_lt18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
  "bmi_30_35", "bmi_35_40", "bmi_ge40"
)
nonreference <- model_categories[model_categories != "bmi_20_25"]
offset_columns <- paste0("offset_logratio_", nonreference)
scale_columns <- c(
  "ntl_within", "pop_density_within", "gdp_within", "pm25_within_aug",
  "urban_share_within", "built_fraction_within", "travel_time_within",
  "temperature_c_within_aug", "relative_humidity_pct_within_aug",
  "elevation_m_within_aug", "age20_39_share_within_aug",
  "age60plus_share_within_aug", "female_adult_share_within_aug"
)
raw_covariate_columns <- c(
  "ntl_log", "pop_density_log", "gdp_log", "pm25",
  "urban_share", "built_fraction", "travel_time_log",
  "temperature_c", "relative_humidity_pct", "elevation_m",
  "age20_39_share", "age60plus_share", "female_adult_share"
)
within_name <- setNames(scale_columns, raw_covariate_columns)

age_groups <- c(
  paste(seq(20, 80, 5), seq(24, 84, 5), sep = "-"),
  "85+"
)
age_display <- c(
  "25-29" = "25-29", "30-34" = "30-34",
  "35-39" = "35-39", "40-44" = "40-44",
  "45-49" = "45-49", "50-54" = "50-54",
  "55-59" = "55-59", "60-64" = "60-64",
  "65-69" = "65-69", "70-74" = "70-74",
  "75-79" = "75-79", "80-84" = "80-84",
  "85+" = "85+"
)
sex_values <- c("Female", "Male")
group_dictionary <- data.table(
  band = 1:16,
  group_label = c(
    "Overall", "Sex:Female", "Sex:Male",
    paste0("Age:", names(age_display))
  ),
  band_name = c(
    "overall_age25plus", "women_age25plus", "men_age25plus",
    paste0("age", gsub("[^0-9]+", "_", names(age_display)), "_both_sexes")
  )
)
sex_age_dictionary <- CJ(
  sex = c("Female", "Male"),
  age = names(age_display),
  unique = TRUE,
  sorted = FALSE
)
sex_age_dictionary[, group_label := paste("SexAge", sex, age, sep = ":")]
sex_age_dictionary[, band_name := sprintf(
  "%s_age%s",
  tolower(sex),
  gsub("[^0-9]+", "_", age)
)]
sex_age_dictionary[, band := .I]
reporting_groups <- rbindlist(
  list(
    group_dictionary[, .(band, group_label, band_name, product = "standard")],
    sex_age_dictionary[, .(
      band, group_label, band_name, product = "sex_age"
    )]
  ),
  use.names = TRUE
)
category_dictionary <- data.table(
  category_order = 1:7,
  model_category = model_categories,
  category = output_categories,
  file_label = c(
    "lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30",
    "30_to_lt35", "35_to_lt40", "ge40"
  ),
  bmi_range = c(
    "<18.5", "18.5 to <20", "20 to <25", "25 to <30",
    "30 to <35", "35 to <40", ">=40"
  )
)
fwrite(
  group_dictionary,
  file.path(metadata_root, "raster_band_dictionary.csv"),
  bom = TRUE
)
fwrite(
  sex_age_dictionary,
  file.path(metadata_root, "sex_age_raster_band_dictionary.csv"),
  bom = TRUE
)
fwrite(
  category_dictionary,
  file.path(metadata_root, "bmi_category_dictionary.csv"),
  bom = TRUE
)

template <- rast(
  xmin = -180, xmax = 180, ymin = -60, ymax = 84,
  resolution = GRID_RESOLUTION, crs = "EPSG:4326"
)

aggregate_to_template <- function(
  raster,
  fun = "mean",
  method = "bilinear",
  target = template
) {
  source_resolution <- res(raster)
  same_crs <- same.crs(raster, target)
  target_resolution <- res(target)
  factor_x <- target_resolution[1] / source_resolution[1]
  factor_y <- target_resolution[2] / source_resolution[2]
  can_aggregate <- same_crs &&
    is.finite(factor_x) && is.finite(factor_y) &&
    abs(factor_x - round(factor_x)) < 1e-5 &&
    abs(factor_y - round(factor_y)) < 1e-5 &&
    factor_x >= 1 && factor_y >= 1
  if (can_aggregate) {
    raster <- aggregate(
      raster,
      fact = c(as.integer(round(factor_x)), as.integer(round(factor_y))),
      fun = fun,
      na.rm = TRUE
    )
    if (!compareGeom(raster, target, stopOnError = FALSE)) {
      raster <- resample(raster, target, method = method)
    }
  } else {
    raster <- project(raster, target, method = method)
  }
  crop(raster, target)
}

write_cache_raster <- function(raster, path, datatype = "FLT4S") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeRaster(
    raster,
    path,
    overwrite = TRUE,
    wopt = list(
      datatype = datatype,
      NAflag = -9999,
      gdal = c(
        "COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
        "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"
      )
    )
  )
  path
}

extract_xy <- function(raster, xy) {
  result <- as.data.table(terra::extract(raster, xy))
  if ("ID" %in% names(result)) result[, ID := NULL]
  result
}

static_covariate_cache <- function(kind, epoch = NA_integer_) {
  suffix <- if (is.finite(epoch)) paste0("_", epoch) else ""
  file.path(
    cache_root,
    "covariates",
    paste0(kind, suffix, "_", RESOLUTION_LABEL, ".tif")
  )
}

build_static_covariate <- function(kind, year) {
  if (kind == "travel_time_log") {
    cache <- static_covariate_cache(kind)
    if (file.exists(cache)) return(rast(cache))
    source <- rast(travel_time_path())
    source[source < 0] <- NA
    result <- aggregate_to_template(log1p(source), fun = "mean")
  } else if (kind == "urban_share") {
    epoch <- smod_year_used(year)
    cache <- static_covariate_cache(kind, epoch)
    if (file.exists(cache)) return(rast(cache))
    source <- rast(smod_path(year))
    source <- ifel(is.na(source), NA, ifel(source >= 21, 1, 0))
    result <- project(source, template, method = "average")
  } else if (kind == "built_fraction") {
    epoch <- builtup_year_used(year)
    cache <- static_covariate_cache(kind, epoch)
    if (file.exists(cache)) return(rast(cache))
    source <- rast(builtup_path(year))
    source <- clamp(
      source / cellSize(source, unit = "m"),
      lower = 0,
      upper = 1,
      values = TRUE
    )
    result <- aggregate_to_template(source, fun = "mean")
  } else {
    stop("Unknown static covariate: ", kind, call. = FALSE)
  }
  names(result) <- kind
  write_cache_raster(result, cache)
  rast(cache)
}

build_annual_covariates <- function(year) {
  cache <- file.path(
    cache_root,
    "covariates",
    sprintf(
      "global_covariates_%d_%s.tif",
      year,
      RESOLUTION_LABEL
    )
  )
  if (file.exists(cache)) return(rast(cache))
  log_line("Combining frozen 0.1-degree covariates for", year)

  base_path <- file.path(
    upstream_cache_root, "covariates",
    sprintf("global_covariates_%d_0p1deg.tif", year)
  )
  climate_path <- file.path(
    climate_cache_root,
    sprintf("ERA5_temperature_humidity_%d_0p1deg.tif", year)
  )
  age_path <- file.path(
    if (year <= 2020L) age_cache_old_root else age_cache_new_root,
    sprintf("WorldPop_adult_age_structure_%d_0p1deg.tif", year)
  )
  for (path in c(base_path, climate_path, age_path, elevation_path)) {
    assert_file(path)
  }

  base <- rast(base_path)
  if (!all(c(
    "ntl_log", "pop_density_log", "gdp_log", "pm25",
    "urban_share", "built_fraction", "travel_time_log"
  ) %in% names(base))) {
    stop("Frozen base covariate stack has unexpected bands", call. = FALSE)
  }
  base <- base[[c(
    "ntl_log", "pop_density_log", "gdp_log", "pm25",
    "urban_share", "built_fraction", "travel_time_log"
  )]]

  climate <- aggregate_to_template(
    rast(climate_path), fun = "mean", method = "bilinear"
  )
  names(climate) <- c("temperature_c", "relative_humidity_pct")
  elevation <- aggregate_to_template(
    rast(elevation_path), fun = "mean", method = "bilinear"
  )
  names(elevation) <- "elevation_m"

  age <- rast(age_path)
  required_age <- c(
    "adult_population", "age20_39_population",
    "age60plus_population", "female_adult_population"
  )
  if (!all(required_age %in% names(age))) {
    stop("WorldPop age structure stack has unexpected bands", call. = FALSE)
  }
  age <- aggregate_to_template(
    age[[required_age]], fun = "sum", method = "near"
  )
  adult <- age[["adult_population"]]
  age20_39_share <- age[["age20_39_population"]] / adult
  age60plus_share <- age[["age60plus_population"]] / adult
  female_adult_share <- age[["female_adult_population"]] / adult
  names(age20_39_share) <- "age20_39_share"
  names(age60plus_share) <- "age60plus_share"
  names(female_adult_share) <- "female_adult_share"

  result <- c(
    base, climate, elevation, age20_39_share,
    age60plus_share, female_adult_share
  )
  names(result) <- raw_covariate_columns
  write_cache_raster(result, cache)
  rm(
    base, climate, elevation, age, adult, age20_39_share,
    age60plus_share, female_adult_share, result
  )
  gc(verbose = FALSE)
  rast(cache)
}

age_population_paths <- function(year, sex, age_group) {
  prefix <- if (sex == "Female") "f" else "m"
  lower <- if (age_group == "85+") c("85", "90") else {
    sprintf("%02d", as.integer(sub("-.*$", "", age_group)))
  }
  file.path(
    PATHS$age_sex_population_dir,
    sprintf(
      "global_%s_%s_%d_CN_1km_R2025A_UA_v1.tif",
      prefix,
      lower,
      year
    )
  )
}

build_age_population <- function(year, sex, age_group) {
  cache <- file.path(
    upstream_cache_root,
    "age_sex_population",
    sprintf(
      "global_%s_%s_%d_%s.tif",
      if (sex == "Female") "f" else "m",
      gsub("[^0-9]+", "_", age_group),
      year,
      RESOLUTION_LABEL
    )
  )
  if (file.exists(cache)) return(rast(cache))
  paths <- age_population_paths(year, sex, age_group)
  missing <- paths[!file.exists(paths)]
  if (length(missing)) {
    stop(
      "Missing WorldPop age-sex raster(s): ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  parts <- lapply(paths, function(path) {
    source <- rast(path)
    source[source < 0] <- NA
    aggregate_to_template(source, fun = "sum", method = "near")
  })
  result <- parts[[1]]
  if (length(parts) > 1L) {
    for (index in 2:length(parts)) result <- result + parts[[index]]
  }
  names(result) <- "population"
  write_cache_raster(result, cache)
  rm(parts, result)
  gc(verbose = FALSE)
  rast(cache)
}

build_country_cells <- function(year) {
  cache <- file.path(
    upstream_cache_root,
    "country_cells",
    sprintf(
      "country_cells_%d_%s.csv.gz",
      year,
      RESOLUTION_LABEL
    )
  )
  if (file.exists(cache)) return(fread(cache))
  dir.create(dirname(cache), recursive = TRUE, showWarnings = FALSE)
  source_dir <- file.path(
    RESULT_ROOT,
    "02_derived",
    sprintf("bmi7_multiscale_%d_country", year)
  )
  files <- list.files(source_dir, pattern = "\\.rds$", full.names = TRUE)
  if (!length(files)) {
    stop("No country grid objects found: ", source_dir, call. = FALSE)
  }
  parts <- lapply(files, function(path) {
    object <- readRDS(path)
    if (
      !is.null(object$error) ||
        is.null(object[[COUNTRY_GRID_MEMBER]])
    ) {
      return(NULL)
    }
    data <- as.data.table(object[[COUNTRY_GRID_MEMBER]])
    data <- data[
      group_label == "Overall" &
        is.finite(population) & population > 0,
      .(iso3, year, x, y, population)
    ]
    unique(data, by = c("iso3", "x", "y"))
  })
  result <- rbindlist(parts, use.names = TRUE, fill = TRUE)
  setorder(result, iso3, y, x)
  result[, row_id := .I]
  fwrite(result, cache)
  result
}

weighted_center_by_country <- function(data, column, weight_column) {
  mean_column <- paste0(column, "_country_mean")
  result_column <- within_name[[column]]
  data[, (mean_column) := weighted_mean_safe(
    get(column),
    get(weight_column)
  ), by = iso3]
  data[!is.finite(get(column)), (column) := get(mean_column)]
  data[, (result_column) := get(column) - get(mean_column)]
  data[, (mean_column) := NULL]
  data
}

posterior_mean <- function(draws, name) {
  value <- draws[[name]]
  apply(value, seq.int(2L, length(dim(value))), mean)
}

recipe_bundle <- readRDS(model_recipe_path)
model_recipe <- recipe_bundle$recipe
model_fit <- readRDS(model_fit_path)
posterior_parameters <- c(
  "beta",
  "z_country_intercept", "z_country_ntl", "z_country_pop",
  "sigma_country_intercept", "sigma_country_ntl", "sigma_country_pop",
  "z_stage_ntl", "z_stage_pop",
  "sigma_stage_ntl", "sigma_stage_pop"
)
posterior <- rstan::extract(
  model_fit,
  pars = posterior_parameters,
  permuted = TRUE
)
beta <- posterior_mean(posterior, "beta")
z_country_intercept <- posterior_mean(posterior, "z_country_intercept")
z_country_ntl <- posterior_mean(posterior, "z_country_ntl")
z_country_pop <- posterior_mean(posterior, "z_country_pop")
z_stage_ntl <- posterior_mean(posterior, "z_stage_ntl")
z_stage_pop <- posterior_mean(posterior, "z_stage_pop")
sigma_country_intercept <- colMeans(posterior$sigma_country_intercept)
sigma_country_ntl <- colMeans(posterior$sigma_country_ntl)
sigma_country_pop <- colMeans(posterior$sigma_country_pop)
sigma_stage_ntl <- colMeans(posterior$sigma_stage_ntl)
sigma_stage_pop <- colMeans(posterior$sigma_stage_pop)

training_countries <- model_recipe$country_levels
training_stages <- model_recipe$stage_levels
ntl_spline <- model_recipe$ntl_spline
pop_spline <- model_recipe$pop_spline

ncd <- fread(ncd_path)
context <- fread(context_path)
context[development_stage == "", development_stage := NA_character_]
latest_context <- context[
  year <= 2023 & !is.na(development_stage),
  .SD[which.max(year)],
  by = iso3
]
latest_context[, year := 2024L]
context <- rbindlist(
  list(context, latest_context[!context, on = .(iso3, year)]),
  use.names = TRUE,
  fill = TRUE
)

scale_value <- function(value, column) {
  (value - model_recipe$scaling[[column]]$center) /
    model_recipe$scaling[[column]]$scale
}

predict_probability <- function(
  grid,
  offset,
  age_midpoint,
  sex,
  country,
  stage
) {
  scaled <- lapply(scale_columns, function(column) {
    scale_value(grid[[column]], column)
  })
  names(scaled) <- scale_columns
  ntl_basis <- predict(ntl_spline, newx = scaled$ntl_within)
  pop_basis <- predict(pop_spline, newx = scaled$pop_density_within)
  female <- as.numeric(sex == "Female")
  age_z <- (age_midpoint - model_recipe$age_scaling$center) /
    model_recipe$age_scaling$scale
  X <- cbind(
    intercept = 1,
    ntl_ns1 = ntl_basis[, 1],
    ntl_ns2 = ntl_basis[, 2],
    ntl_ns3 = ntl_basis[, 3],
    pop_ns1 = pop_basis[, 1],
    pop_ns2 = pop_basis[, 2],
    pop_ns3 = pop_basis[, 3],
    gdp_within = scaled$gdp_within,
    pm25_within_aug = scaled$pm25_within_aug,
    urban_share_within = scaled$urban_share_within,
    built_fraction_within = scaled$built_fraction_within,
    travel_time_within = scaled$travel_time_within,
    temperature_c_within_aug = scaled$temperature_c_within_aug,
    relative_humidity_pct_within_aug =
      scaled$relative_humidity_pct_within_aug,
    elevation_m_within_aug = scaled$elevation_m_within_aug,
    age20_39_share_within_aug = scaled$age20_39_share_within_aug,
    age60plus_share_within_aug = scaled$age60plus_share_within_aug,
    female_adult_share_within_aug = scaled$female_adult_share_within_aug,
    ntl_by_female = scaled$ntl_within * female,
    pop_by_female = scaled$pop_density_within * female,
    ntl_by_age = scaled$ntl_within * age_z,
    pop_by_age = scaled$pop_density_within * age_z
  )
  if (!identical(colnames(X), recipe_bundle$feature_names)) {
    stop("Stage 9 design matrix does not match frozen Stage 3 feature order")
  }
  eta <- matrix(
    rep(offset, each = nrow(grid)),
    nrow = nrow(grid),
    byrow = FALSE
  ) + X %*% beta

  country_index <- match(country, training_countries)
  stage_index <- match(stage, training_stages)
  for (q in seq_along(nonreference)) {
    if (is.finite(country_index)) {
      eta[, q] <- eta[, q] +
        sigma_country_intercept[q] *
          z_country_intercept[q, country_index] +
        scaled$ntl_within *
          sigma_country_ntl[q] *
          z_country_ntl[q, country_index] +
        scaled$pop_density_within *
          sigma_country_pop[q] *
          z_country_pop[q, country_index]
    }
    if (is.finite(stage_index)) {
      eta[, q] <- eta[, q] +
        scaled$ntl_within *
          sigma_stage_ntl[q] *
          z_stage_ntl[q, stage_index] +
        scaled$pop_density_within *
          sigma_stage_pop[q] *
          z_stage_pop[q, stage_index]
    }
  }

  eta_full <- matrix(0, nrow(grid), length(model_categories))
  eta_full[, model_categories != "bmi_20_25"] <- eta
  row_maximum <- apply(eta_full, 1, max)
  probability <- exp(eta_full - row_maximum)
  probability / rowSums(probability)
}

rake_composition <- function(probability, weight, target) {
  valid <- is.finite(weight) & weight > 0 &
    rowSums(is.finite(probability)) == ncol(probability)
  if (!any(valid)) return(probability)
  target <- pmax(target, 1e-10)
  target <- target / sum(target)
  for (iteration in seq_len(200L)) {
    current <- colSums(
      probability[valid, , drop = FALSE] * weight[valid]
    ) / sum(weight[valid])
    factor <- target / pmax(current, 1e-12)
    probability[valid, ] <- sweep(
      probability[valid, , drop = FALSE],
      2,
      factor,
      "*"
    )
    probability[valid, ] <- probability[valid, , drop = FALSE] /
      rowSums(probability[valid, , drop = FALSE])
    if (max(abs(current - target)) < 1e-10) break
  }
  probability
}

add_group_counts <- function(accumulator, label, indices, count, population) {
  if (!label %in% names(accumulator)) {
    accumulator[[label]] <- list(
      population = numeric(length(population)),
      count = matrix(0, length(population), length(model_categories))
    )
  }
  accumulator[[label]]$population[indices] <-
    accumulator[[label]]$population[indices] + population[indices]
  accumulator[[label]]$count[indices, ] <-
    accumulator[[label]]$count[indices, , drop = FALSE] +
    count[indices, , drop = FALSE]
  accumulator
}

make_basis_points <- function(data) {
  data <- copy(data)
  previous <- integer(nrow(data))
  cumulative <- numeric(nrow(data))
  for (index in seq_len(6L)) {
    category <- output_categories[index]
    cumulative <- cumulative + data[[category]]
    current <- as.integer(round(
      10000 * cumulative / data$population
    ))
    current <- pmax(previous, pmin(10000L, current))
    data[, paste0(category, "_bp") := current - previous]
    previous <- current
  }
  data[, paste0(output_categories[7], "_bp") := 10000L - previous]
  data
}

make_stack <- function(data, value_column, dictionary) {
  layers <- vector("list", nrow(dictionary))
  for (index in seq_len(nrow(dictionary))) {
    label <- dictionary$group_label[index]
    part <- data[group_label == label & is.finite(get(value_column))]
    layer <- template
    if (nrow(part)) {
      cells <- cellFromXY(layer, as.matrix(part[, .(x, y)]))
      valid <- is.finite(cells)
      layer[cells[valid]] <- part[[value_column]][valid]
    }
    names(layer) <- dictionary$band_name[index]
    layers[[index]] <- layer
  }
  result <- rast(layers)
  names(result) <- dictionary$band_name
  result
}

write_bmi_stack <- function(stack, path) {
  units(stack) <- rep(
    "basis points; 10000 equals 100 percent",
    nlyr(stack)
  )
  writeRaster(
    stack,
    path,
    overwrite = TRUE,
    wopt = list(
      datatype = "INT2U",
      NAflag = 65535,
      gdal = c(
        "COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
        "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"
      )
    )
  )
}

qa_path <- file.path(
  metadata_root,
  "regional_steps_mrp_raster_export_QA.csv"
)
qa_existing <- if (file.exists(qa_path)) {
  fread(qa_path)
} else {
  data.table()
}
qa_parts <- list()
for (year in YEARS) {
  current_year <- year
  year_output <- file.path(output_root, as.character(year))
  dir.create(year_output, recursive = TRUE, showWarnings = FALSE)
  complete_flag <- file.path(year_output, "COMPLETE.flag")
  if (file.exists(complete_flag) && !overwrite) {
    log_line("Skipping completed year", year)
    next
  }
  if (nrow(qa_existing)) {
    qa_existing <- qa_existing[year != current_year]
  }

  log_line("Preparing country cells for", year)
  grid <- build_country_cells(year)
  annual_covariates <- build_annual_covariates(year)
  covariate_values <- extract_xy(
    annual_covariates,
    as.matrix(grid[, .(x, y)])
  )
  grid <- cbind(grid, covariate_values)
  missing_before <- vapply(
    raw_covariate_columns,
    function(column) mean(!is.finite(grid[[column]])),
    numeric(1)
  )
  for (column in raw_covariate_columns) {
    grid <- weighted_center_by_country(grid, column, "population")
    grid[!is.finite(get(within_name[[column]])), (within_name[[column]]) := 0]
  }

  log_line("Aggregating WorldPop age-sex rasters for", year)
  population_matrix <- matrix(
    0,
    nrow = nrow(grid),
    ncol = length(age_groups) * length(sex_values)
  )
  population_labels <- character(ncol(population_matrix))
  position <- 0L
  for (sex in sex_values) {
    for (age_group in age_groups) {
      position <- position + 1L
      age_raster <- build_age_population(year, sex, age_group)
      extracted_population <- extract_xy(
        age_raster,
        as.matrix(grid[, .(x, y)])
      )
      population_matrix[, position] <- as.numeric(
        extracted_population[[1]]
      )
      population_labels[position] <- paste(sex, age_group, sep = "|")
      rm(age_raster)
    }
  }
  population_matrix[!is.finite(population_matrix)] <- 0
  raw_population <- rowSums(population_matrix)
  scale_factor <- grid$population / raw_population
  scale_factor[!is.finite(scale_factor) | scale_factor < 0] <- NA_real_

  # Border and small-island cells can miss the centre of the global raster.
  # Their age-sex composition is imputed from the remaining cells of the same
  # country, then rescaled to the clipped adult population in the old grid.
  missing_population <- !is.finite(scale_factor)
  if (any(missing_population)) {
    for (iso3 in unique(grid$iso3[missing_population])) {
      country_rows <- which(grid$iso3 == iso3)
      valid_rows <- country_rows[raw_population[country_rows] > 0]
      if (length(valid_rows)) {
        national_mix <- colSums(
          population_matrix[valid_rows, , drop = FALSE]
        )
        national_mix <- national_mix / sum(national_mix)
      } else {
        national_mix <- rep(1 / ncol(population_matrix), ncol(population_matrix))
      }
      rows <- country_rows[missing_population[country_rows]]
      population_matrix[rows, ] <- grid$population[rows] * rep(
        national_mix,
        each = length(rows)
      )
      scale_factor[rows] <- 1
    }
  }
  population_matrix <- population_matrix * scale_factor

  accumulators <- list()
  anchor_qa <- list()
  stratum_index <- 0L
  for (iso_value in unique(grid$iso3)) {
    year_value <- year
    country_rows <- which(grid$iso3 == iso_value)
    country_context <- context[
      iso3 == iso_value & year == year_value
    ][1]
    stage <- if (nrow(country_context)) {
      as.character(country_context$development_stage)
    } else {
      NA_character_
    }
    for (sex in sex_values) {
      for (age_group in age_groups) {
        if (age_group == "20-24") next
        stratum_index <- stratum_index + 1L
        column <- match(
          paste(sex, age_group, sep = "|"),
          population_labels
        )
        weights <- population_matrix[country_rows, column]
        if (!any(is.finite(weights) & weights > 0)) next
        sex_value <- sex
        age_group_value <- age_group
        target_row <- ncd[
          iso3 == iso_value & year == year_value &
            sex == sex_value & age_group == age_group_value
        ][1]
        if (!nrow(target_row)) next
        target <- as.numeric(target_row[, ..output_categories])
        if (any(!is.finite(target)) || sum(target) <= 0) next
        target <- target / sum(target)
        offset <- log(
          pmax(target[model_categories != "bmi_20_25"], 1e-10) /
            pmax(target[model_categories == "bmi_20_25"], 1e-10)
        )
        lower <- if (age_group == "85+") {
          85L
        } else {
          as.integer(sub("-.*$", "", age_group))
        }
        age_midpoint <- if (age_group == "85+") 90 else lower + 2
        probability <- predict_probability(
          grid[country_rows],
          offset,
          age_midpoint,
          sex,
          iso_value,
          stage
        )
        probability <- rake_composition(probability, weights, target)
        counts <- probability * weights

        global_rows <- country_rows
        labels <- c(
          "Overall",
          paste0("Sex:", sex),
          paste0("Age:", age_display[[age_group]]),
          paste0(
            "SexAge:", sex, ":", age_display[[age_group]]
          )
        )
        for (label in labels) {
          if (!label %in% names(accumulators)) {
            accumulators[[label]] <- list(
              population = numeric(nrow(grid)),
              count = matrix(
                0,
                nrow(grid),
                length(model_categories)
              )
            )
          }
          accumulators[[label]]$population[global_rows] <-
            accumulators[[label]]$population[global_rows] + weights
          accumulators[[label]]$count[global_rows, ] <-
            accumulators[[label]]$count[global_rows, , drop = FALSE] +
            counts
        }

        realized <- colSums(counts) / sum(weights)
        anchor_qa[[length(anchor_qa) + 1L]] <- data.table(
          iso3 = iso_value,
          year = year,
          development_stage = stage,
          trained_country = iso_value %chin% training_countries,
          trained_development_stage = stage %chin% training_stages,
          sex = sex,
          age_group = age_group,
          category = output_categories,
          target = target,
          realized = realized,
          absolute_error = abs(target - realized)
        )
      }
    }
  }

  global_parts <- lapply(reporting_groups$group_label, function(label) {
    accumulator <- accumulators[[label]]
    result <- data.table(
      x = grid$x,
      y = grid$y,
      iso3 = grid$iso3,
      year = year,
      group_label = label,
      population = accumulator$population
    )
    for (index in seq_along(output_categories)) {
      result[, (output_categories[index]) := accumulator$count[, index]]
    }
    result <- result[is.finite(population) & population > 0]
    result[, lapply(.SD, sum, na.rm = TRUE),
      by = .(year, group_label, x, y),
      .SDcols = c("population", output_categories)
    ]
  })
  global_grid <- rbindlist(
    global_parts,
    use.names = TRUE,
    fill = TRUE
  )
  global_grid <- make_basis_points(global_grid)

  prediction_object <- list(
    year = year,
    resolution = GRID_RESOLUTION,
    model = paste(
      "Enhanced Bayesian multinomial logistic-normal MRP selected",
      "by the frozen Stage 7 conditional mapping gate"
    ),
    model_fit_path = normalizePath(
      model_fit_path, winslash = "/", mustWork = TRUE
    ),
    model_recipe_path = normalizePath(
      model_recipe_path, winslash = "/", mustWork = TRUE
    ),
    global_grid = global_grid,
    anchor_qa = rbindlist(anchor_qa, use.names = TRUE, fill = TRUE),
    covariate_missing_before_country_imputation = missing_before,
    training_countries = training_countries,
    training_development_stages = training_stages
  )
  saveRDS(
    prediction_object,
    file.path(
      derived_root,
      sprintf(
        "global_%d_bmi7_regional_steps_mrp_%s.rds",
        year,
        RESOLUTION_LABEL
      )
    ),
    compress = if (GRID_RESOLUTION == 0.1) "gzip" else "xz"
  )

  population_standard <- make_stack(
    global_grid,
    "population",
    group_dictionary
  )
  population_sex_age <- make_stack(
    global_grid,
    "population",
    sex_age_dictionary
  )
  display_mask <- ifel(
    is.finite(population_standard[[1]]) &
      population_standard[[1]] >= 100,
    1,
    NA
  )
  population_standard <- mask(population_standard, display_mask)
  population_sex_age <- mask(population_sex_age, display_mask)

  for (index in seq_len(nrow(category_dictionary))) {
    category <- category_dictionary$category[index]
    value_column <- paste0(category, "_bp")
    stack <- make_stack(
      global_grid,
      value_column,
      group_dictionary
    )
    stack <- mask(stack, display_mask)
    output_path <- file.path(
      year_output,
      sprintf(
        "global_%d_bmi_%s_regional_steps_mrp_%s_bp.tif",
        year,
        category_dictionary$file_label[index],
        RESOLUTION_LABEL
      )
    )
    write_bmi_stack(stack, output_path)
    written <- rast(output_path)
    qa_parts[[length(qa_parts) + 1L]] <- data.table(
      year = year,
      category = category,
      file = normalizePath(output_path, winslash = "/", mustWork = TRUE),
      file_size_bytes = file.info(output_path)$size,
      bands = nlyr(written),
      rows = nrow(written),
      columns = ncol(written),
      composition_anchor_max_error = max(
        prediction_object$anchor_qa$absolute_error,
        na.rm = TRUE
      ),
      qa_pass = nlyr(written) == 16L &&
        nrow(written) == nrow(template) &&
        ncol(written) == ncol(template) &&
      identical(names(written), group_dictionary$band_name)
    )
    rm(stack, written)

    sex_age_stack <- make_stack(
      global_grid,
      value_column,
      sex_age_dictionary
    )
    sex_age_stack <- mask(sex_age_stack, display_mask)
    sex_age_output_path <- file.path(
      year_output,
      sprintf(
        "global_%d_bmi_%s_regional_steps_mrp_sex_age_%s_bp.tif",
        year,
        category_dictionary$file_label[index],
        RESOLUTION_LABEL
      )
    )
    write_bmi_stack(sex_age_stack, sex_age_output_path)
    written_sex_age <- rast(sex_age_output_path)
    qa_parts[[length(qa_parts) + 1L]] <- data.table(
      year = year,
      category = category,
      product = "sex_age",
      file = normalizePath(
        sex_age_output_path,
        winslash = "/",
        mustWork = TRUE
      ),
      file_size_bytes = file.info(sex_age_output_path)$size,
      bands = nlyr(written_sex_age),
      rows = nrow(written_sex_age),
      columns = ncol(written_sex_age),
      composition_anchor_max_error = max(
        prediction_object$anchor_qa$absolute_error,
        na.rm = TRUE
      ),
      qa_pass = nlyr(written_sex_age) == 26L &&
        nrow(written_sex_age) == nrow(template) &&
        ncol(written_sex_age) == ncol(template) &&
        identical(
          names(written_sex_age),
          sex_age_dictionary$band_name
        )
    )
    qa_parts[[length(qa_parts) - 1L]][, product := "standard"]
    rm(sex_age_stack, written_sex_age)
  }

  population_standard_path <- file.path(
    year_output,
    sprintf(
      "global_%d_adult_population_regional_steps_mrp_%s.tif",
      year,
      RESOLUTION_LABEL
    )
  )
  write_cache_raster(
    population_standard,
    population_standard_path
  )
  population_sex_age_path <- file.path(
    year_output,
    sprintf(
      "global_%d_adult_population_regional_steps_mrp_sex_age_%s.tif",
      year,
      RESOLUTION_LABEL
    )
  )
  write_cache_raster(
    population_sex_age,
    population_sex_age_path
  )
  rm(population_standard, population_sex_age, display_mask)

  fwrite(
    prediction_object$anchor_qa,
    file.path(
      metadata_root,
      sprintf("country_age_sex_anchor_QA_%d.csv", year)
    ),
    bom = TRUE
  )
  writeLines("COMPLETE", complete_flag, useBytes = TRUE)
  fwrite(
    rbindlist(
      c(list(qa_existing), qa_parts),
      use.names = TRUE,
      fill = TRUE
    ),
    qa_path,
    bom = TRUE
  )
  log_line(
    "Completed", year,
    "maximum anchor error",
    signif(max(prediction_object$anchor_qa$absolute_error), 3)
  )
  rm(
    grid, annual_covariates, covariate_values, population_matrix,
    accumulators, global_grid, prediction_object
  )
  gc(verbose = FALSE)
  tmpFiles(remove = TRUE, old = FALSE)
}

qa <- rbindlist(
  c(list(qa_existing), qa_parts),
  use.names = TRUE,
  fill = TRUE
)
if (nrow(qa)) {
  if (any(!qa$qa_pass) || any(qa$file_size_bytes <= 0)) {
    stop("Regional STEPS MRP raster export QA failed", call. = FALSE)
  }
}
writeLines(
  "QA_PASS",
  file.path(metadata_root, "regional_steps_mrp_raster_export_QA.flag"),
  useBytes = TRUE
)
log_line("Completed enhanced-MRP BMI7 0.1-degree grid prediction")

