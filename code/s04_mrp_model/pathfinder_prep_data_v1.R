# Prepare Stan JSON data for the Pathfinder pilot.
# Rebuilds the exact design matrix of full_covariate_mrp_refit_v1_20260905
# (same training table, same recipe/scaling code) and writes cmdstan JSON.
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
out_json <- if (length(args) >= 1) args[[1]] else
  file.path(.cfg$CMDSTAN_WORK, "bmi7_pilot", "data.json")

refit_root <- file.path(.cfg$RESULTS, "04_future_capable_mrp", "full_covariate_mrp_refit_v1_20260905")
training_path <- file.path(refit_root, "data/bmi7_full_covariate_training_v1.csv.gz")

bmi_labels <- c(
  "bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
  "bmi_30_35", "bmi_35_40", "bmi_ge_40"
)
nonreference_labels <- bmi_labels[bmi_labels != "bmi_20_25"]
count_columns <- paste0("count_", bmi_labels)
offset_columns <- paste0("offset_logratio_", nonreference_labels)
scaled_columns <- c(
  "ntl_within", "pop_density_within", "gdp_within", "pm25_within_aug",
  "urban_share_within", "built_fraction_within", "travel_time_within",
  "temperature_c_within_aug", "relative_humidity_pct_within_aug",
  "elevation_m_within_aug", "age20_39_share_within_aug",
  "age60plus_share_within_aug", "female_adult_share_within_aug"
)

weighted_center_scale <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  center <- sum(value[keep] * weight[keep]) / sum(weight[keep])
  scale <- sqrt(sum(weight[keep] * (value[keep] - center)^2) / sum(weight[keep]))
  if (!is.finite(scale) || scale < 1e-8) scale <- 1
  list(center = center, scale = scale)
}
apply_scaling <- function(value, specification) {
  (value - specification$center) / specification$scale
}

data <- fread(training_path, showProgress = FALSE)
weight <- as.numeric(data$pseudo_n)
scaling <- lapply(scaled_columns, function(column) {
  weighted_center_scale(as.numeric(data[[column]]), weight)
})
names(scaling) <- scaled_columns
age_scaling <- weighted_center_scale(as.numeric(data$age_midpoint), weight)
scaled <- lapply(scaled_columns, function(column) {
  apply_scaling(as.numeric(data[[column]]), scaling[[column]])
})
names(scaled) <- scaled_columns
ntl_basis <- predict(ns(scaled$ntl_within, df = 3), newx = scaled$ntl_within)
pop_basis <- predict(ns(scaled$pop_density_within, df = 3),
                     newx = scaled$pop_density_within)
age_scaled <- apply_scaling(data$age_midpoint, age_scaling)
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
stopifnot(ncol(X) == 22L, all(is.finite(X)))

stan_data <- list(
  N = nrow(data), K = length(bmi_labels), P = ncol(X),
  C = length(sort(unique(data$iso3))),
  D = length(sort(unique(as.character(data$development_stage)))),
  F = length(sort(unique(data$source_family))),
  S = length(sort(unique(data$survey_id))),
  country_id = match(data$iso3, sort(unique(data$iso3))),
  stage_id = match(as.character(data$development_stage),
                   sort(unique(as.character(data$development_stage)))),
  family_id = match(data$source_family, sort(unique(data$source_family))),
  survey_id = match(data$survey_id, sort(unique(data$survey_id))),
  is_train = rep(1L, nrow(data)),
  y = as.matrix(data[, ..count_columns]),
  alr_offset = as.matrix(data[, ..offset_columns]),
  X = X,
  ntl_z = scaled$ntl_within,
  pop_z = scaled$pop_density_within
)
stopifnot(!anyNA(stan_data$country_id), !anyNA(stan_data$stage_id),
          !anyNA(stan_data$family_id), !anyNA(stan_data$survey_id))

# enforce integer typing for Stan JSON
stan_data$N <- as.integer(stan_data$N); stan_data$K <- as.integer(stan_data$K)
stan_data$P <- as.integer(stan_data$P); stan_data$C <- as.integer(stan_data$C)
stan_data$D <- as.integer(stan_data$D); stan_data$F <- as.integer(stan_data$F)
stan_data$S <- as.integer(stan_data$S)
stan_data$country_id <- as.integer(stan_data$country_id)
stan_data$stage_id <- as.integer(stan_data$stage_id)
stan_data$family_id <- as.integer(stan_data$family_id)
stan_data$survey_id <- as.integer(stan_data$survey_id)
stan_data$is_train <- as.integer(stan_data$is_train)
stan_data$y <- matrix(as.integer(stan_data$y), nrow = nrow(stan_data$y))

dir.create(dirname(out_json), recursive = TRUE, showWarnings = FALSE)
write_json(stan_data, out_json, auto_unbox = TRUE, digits = NA,
           na = "null", pretty = FALSE)
cat("wrote", out_json, file.size(out_json), "bytes; N =", stan_data$N,
    "P =", stan_data$P, "\n")
