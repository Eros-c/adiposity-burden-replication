if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

suppressPackageStartupMessages({
  library(data.table)
  library(digest)
  library(jsonlite)
  library(rstan)
  library(splines)
})

rstan_options(auto_write = FALSE)
options(mc.cores = 1L)

stage_root <- file.path(.cfg$RESULTS, "05_historical_bmi_1990_2024", "phase5_temporal_backtesting_v1", "baseline_full_mrp")
training_path <- file.path(.cfg$ROOT, "results", "bmi_downscaling_ascii", "experiments", "BMI7_MRP_STEPS_regional_augmentation_20260812", "data", "bmi7_enhanced_plus_steps_regional_training_v1.csv.gz")
assignment_path <- file.path(.cfg$ROOT, "results", "bmi_downscaling_ascii", "experiments", "BMI7_MRP_STEPS_regional_augmentation_20260812", "validation_design", "all_eligible20_v1", "all_eligible20_validation_assignments_v1.csv.gz")
model_path <- file.path(.cfg$ROOT, "results", "bmi_downscaling_ascii", "03_models", "bmi7_logistic_normal_mrp_auxiliary_stanmodel.rds")
stan_source_path <- file.path(.cfg$ROOT, "scripts", "bmi_downscaling", "stan", "bmi7_logistic_normal_mrp_auxiliary.stan")

base_seed <- 20260829L
training_only_countries <- c("CHN", "JPN")
bmi_labels <- c(
  "bmi_lt_18_5", "bmi_18_5_20", "bmi_20_25", "bmi_25_30",
  "bmi_30_35", "bmi_35_40", "bmi_ge_40"
)
nonreference_labels <- bmi_labels[bmi_labels != "bmi_20_25"]
count_columns <- paste0("count_", bmi_labels)
offset_columns <- paste0("offset_logratio_", nonreference_labels)
support <- c(17.0, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5)

scaled_columns <- c(
  "ntl_within",
  "pop_density_within",
  "gdp_within",
  "pm25_within_aug",
  "urban_share_within",
  "built_fraction_within",
  "travel_time_within",
  "temperature_c_within_aug",
  "relative_humidity_pct_within_aug",
  "elevation_m_within_aug",
  "age20_39_share_within_aug",
  "age60plus_share_within_aug",
  "female_adult_share_within_aug"
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: phase5_baseline_temporal_task_v1.R <task_id>", call. = FALSE)
}
task_id <- args[[1]]

dir.create(stage_root, recursive = TRUE, showWarnings = FALSE)
for (subdir in c(
  "data", "models", "models/checkpoints", "validation",
  "validation/predictions", "validation/metrics", "validation/by_source",
  "qa", "logs"
)) {
  dir.create(file.path(stage_root, subdir), recursive = TRUE, showWarnings = FALSE)
}

write_json_atomic <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  write_json(value, temporary, pretty = TRUE, auto_unbox = TRUE, na = "null")
  if (file.exists(path)) file.remove(path)
  if (!file.rename(temporary, path)) {
    stop("Could not atomically write JSON: ", path, call. = FALSE)
  }
}

task_seed <- function(text) {
  values <- utf8ToInt(enc2utf8(text))
  as.integer(base_seed + sum(values * seq_along(values)) %% 100000L)
}

weighted_center_scale <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  if (!any(keep)) stop("No finite values for weighted scaling", call. = FALSE)
  center <- sum(value[keep] * weight[keep]) / sum(weight[keep])
  scale <- sqrt(sum(weight[keep] * (value[keep] - center)^2) / sum(weight[keep]))
  if (!is.finite(scale) || scale < 1e-8) scale <- 1
  list(center = center, scale = scale)
}

apply_scaling <- function(value, specification) {
  (value - specification$center) / specification$scale
}

close_composition <- function(matrix_value, epsilon = 1e-12) {
  output <- pmax(as.matrix(matrix_value), epsilon)
  output / rowSums(output)
}

weighted_mean_safe <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  if (!any(keep)) return(NA_real_)
  sum(value[keep] * weight[keep]) / sum(weight[keep])
}

weighted_r2 <- function(observed, predicted, weight) {
  keep <- is.finite(observed) & is.finite(predicted) &
    is.finite(weight) & weight > 0
  if (sum(keep) < 2L) return(NA_real_)
  observed <- observed[keep]
  predicted <- predicted[keep]
  weight <- weight[keep]
  center <- weighted_mean_safe(observed, weight)
  denominator <- sum(weight * (observed - center)^2)
  if (!is.finite(denominator) || denominator <= 0) return(NA_real_)
  1 - sum(weight * (observed - predicted)^2) / denominator
}

weighted_correlation <- function(x, y, weight) {
  keep <- is.finite(x) & is.finite(y) & is.finite(weight) & weight > 0
  if (sum(keep) < 4L) return(NA_real_)
  x <- x[keep]
  y <- y[keep]
  weight <- weight[keep]
  x_center <- weighted_mean_safe(x, weight)
  y_center <- weighted_mean_safe(y, weight)
  covariance <- weighted_mean_safe(
    (x - x_center) * (y - y_center), weight
  )
  x_sd <- sqrt(weighted_mean_safe((x - x_center)^2, weight))
  y_sd <- sqrt(weighted_mean_safe((y - y_center)^2, weight))
  if (!is.finite(x_sd) || !is.finite(y_sd) || x_sd <= 0 || y_sd <= 0) {
    return(NA_real_)
  }
  covariance / (x_sd * y_sd)
}

row_wasserstein2 <- function(observed, predicted) {
  observed <- close_composition(observed)
  predicted <- close_composition(predicted)
  output <- numeric(nrow(observed))
  for (row_index in seq_len(nrow(observed))) {
    boundaries <- sort(unique(c(
      0, 1, cumsum(observed[row_index, ]), cumsum(predicted[row_index, ])
    )))
    boundaries <- pmin(1, pmax(0, boundaries))
    middle <- (head(boundaries, -1L) + tail(boundaries, -1L)) / 2
    width <- diff(boundaries)
    observed_index <- findInterval(
      middle, c(0, cumsum(observed[row_index, ])),
      rightmost.closed = TRUE, all.inside = TRUE
    )
    predicted_index <- findInterval(
      middle, c(0, cumsum(predicted[row_index, ])),
      rightmost.closed = TRUE, all.inside = TRUE
    )
    observed_index <- pmin(length(support), pmax(1L, observed_index))
    predicted_index <- pmin(length(support), pmax(1L, predicted_index))
    output[row_index] <- sqrt(sum(
      width * (support[observed_index] - support[predicted_index])^2
    ))
  }
  output
}

posterior_mean <- function(draws, name) {
  value <- draws[[name]]
  apply(value, seq.int(2L, length(dim(value))), mean)
}

load_inputs <- function() {
  required <- c(training_path, assignment_path, model_path, stan_source_path)
  missing <- required[!file.exists(required)]
  if (length(missing)) {
    stop("Missing Stage 3 input: ", missing[[1]], call. = FALSE)
  }
  data <- fread(training_path, showProgress = FALSE)
  assignments <- fread(assignment_path, showProgress = FALSE)
  if (nrow(data) != nrow(assignments)) {
    stop("Training and assignment row counts differ", call. = FALSE)
  }
  if (anyDuplicated(data$row_id_stage01) ||
      anyDuplicated(assignments$row_id_stage01)) {
    stop("Duplicate row_id_stage01", call. = FALSE)
  }
  setkey(data, row_id_stage01)
  setkey(assignments, row_id_stage01)
  if (!identical(data$row_id_stage01, assignments$row_id_stage01)) {
    stop("Training and validation-assignment row IDs do not align", call. = FALSE)
  }
  list(data = data, assignments = assignments)
}

resolve_task <- function(task, data, assignments) {
  all_rows <- data$row_id_stage01
  temporal_specs <- list(
    time_cut2014_test2015_2017 = c(2014L, 2015L, 2017L),
    time_cut2017_test2018_2020 = c(2017L, 2018L, 2020L),
    time_cut2020_test2021_2024 = c(2020L, 2021L, 2024L)
  )
  if (task %in% names(temporal_specs)) {
    specification <- temporal_specs[[task]]
    cutoff <- specification[[1]]
    test_start <- specification[[2]]
    test_end <- specification[[3]]
    eligible_ids <- assignments[
      eligible_validation == TRUE,
      row_id_stage01
    ]
    train_ids <- data[year <= cutoff, row_id_stage01]
    test_ids <- data[
      row_id_stage01 %in% eligible_ids & between(year, test_start, test_end),
      row_id_stage01
    ]
    return(list(
      task_id = task,
      scheme = "forward_temporal_holdout",
      fold = paste0(test_start, "-", test_end),
      train_ids = train_ids,
      test_ids = test_ids,
      excluded_ids = setdiff(all_rows, c(train_ids, test_ids)),
      train_cutoff_year = cutoff,
      test_start_year = test_start,
      test_end_year = test_end
    ))
  }
  if (task == "full_model") {
    return(list(
      task_id = task,
      scheme = "full_model",
      fold = "all",
      train_ids = all_rows,
      test_ids = integer(),
      excluded_ids = integer()
    ))
  }
  if (task == "smoke_preflight") {
    candidate_test <- assignments[
      eligible_validation == TRUE & internal_fold == 1L,
      row_id_stage01
    ]
    candidate_train <- setdiff(all_rows, candidate_test)
    train_ids <- candidate_train[
      order((candidate_train * 2654435761) %% 2147483647)
    ][seq_len(min(3000L, length(candidate_train)))]
    test_ids <- candidate_test[
      order((candidate_test * 11400714819323198485) %% 2147483647)
    ][seq_len(min(500L, length(candidate_test)))]
    return(list(
      task_id = task,
      scheme = "smoke_preflight",
      fold = "1",
      train_ids = train_ids,
      test_ids = test_ids,
      excluded_ids = integer()
    ))
  }
  if (grepl("^internal_fold_[1-5]$", task)) {
    fold <- as.integer(sub("^internal_fold_", "", task))
    test_ids <- assignments[
      eligible_validation == TRUE & internal_fold == fold,
      row_id_stage01
    ]
    return(list(
      task_id = task,
      scheme = "all_eligible20_internal_random_row_5fold",
      fold = as.character(fold),
      train_ids = setdiff(all_rows, test_ids),
      test_ids = test_ids,
      excluded_ids = integer()
    ))
  }
  if (grepl("^country_fold_[1-5]$", task)) {
    fold <- as.integer(sub("^country_fold_", "", task))
    held_out_countries <- assignments[
      eligible_validation == TRUE & country_fold == fold,
      unique(iso3)
    ]
    test_ids <- assignments[
      eligible_validation == TRUE & country_fold == fold,
      row_id_stage01
    ]
    excluded_ids <- data[
      iso3 %in% held_out_countries & !row_id_stage01 %in% test_ids,
      row_id_stage01
    ]
    return(list(
      task_id = task,
      scheme = "all_eligible20_country_grouped_5fold",
      fold = as.character(fold),
      train_ids = data[!iso3 %in% held_out_countries, row_id_stage01],
      test_ids = test_ids,
      excluded_ids = excluded_ids,
      held_out_countries = sort(held_out_countries)
    ))
  }
  stop("Unknown unified validation task: ", task, call. = FALSE)
}

build_recipe <- function(train) {
  weight <- as.numeric(train$pseudo_n)
  scaling <- lapply(scaled_columns, function(column) {
    weighted_center_scale(as.numeric(train[[column]]), weight)
  })
  names(scaling) <- scaled_columns
  age_scaling <- weighted_center_scale(as.numeric(train$age_midpoint), weight)
  ntl_scaled <- apply_scaling(train$ntl_within, scaling$ntl_within)
  pop_scaled <- apply_scaling(
    train$pop_density_within, scaling$pop_density_within
  )
  list(
    scaling = scaling,
    age_scaling = age_scaling,
    ntl_spline = ns(ntl_scaled, df = 3),
    pop_spline = ns(pop_scaled, df = 3),
    country_levels = sort(unique(train$iso3)),
    stage_levels = sort(unique(as.character(train$development_stage))),
    family_levels = sort(unique(train$source_family)),
    survey_levels = sort(unique(train$survey_id))
  )
}

make_design <- function(data, recipe) {
  scaled <- lapply(scaled_columns, function(column) {
    apply_scaling(as.numeric(data[[column]]), recipe$scaling[[column]])
  })
  names(scaled) <- scaled_columns
  ntl_basis <- predict(recipe$ntl_spline, newx = scaled$ntl_within)
  pop_basis <- predict(recipe$pop_spline, newx = scaled$pop_density_within)
  age_scaled <- apply_scaling(data$age_midpoint, recipe$age_scaling)
  female <- as.numeric(data$sex == "Female")
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
    ntl_by_age = scaled$ntl_within * age_scaled,
    pop_by_age = scaled$pop_density_within * age_scaled
  )
  if (any(!is.finite(X))) stop("Non-finite design matrix", call. = FALSE)
  list(
    X = X,
    ntl_z = scaled$ntl_within,
    pop_z = scaled$pop_density_within,
    country_id = match(data$iso3, recipe$country_levels),
    stage_id = match(as.character(data$development_stage), recipe$stage_levels),
    family_id = match(data$source_family, recipe$family_levels),
    survey_id = match(data$survey_id, recipe$survey_levels)
  )
}

make_stan_data <- function(train, recipe, design) {
  if (anyNA(design$country_id) || anyNA(design$stage_id) ||
      anyNA(design$family_id) || anyNA(design$survey_id)) {
    stop("Training levels were not mapped", call. = FALSE)
  }
  list(
    N = nrow(train),
    K = length(bmi_labels),
    P = ncol(design$X),
    C = length(recipe$country_levels),
    D = length(recipe$stage_levels),
    F = length(recipe$family_levels),
    S = length(recipe$survey_levels),
    country_id = design$country_id,
    stage_id = design$stage_id,
    family_id = design$family_id,
    survey_id = design$survey_id,
    is_train = rep(1L, nrow(train)),
    y = as.matrix(train[, ..count_columns]),
    offset = as.matrix(train[, ..offset_columns]),
    X = design$X,
    ntl_z = design$ntl_z,
    pop_z = design$pop_z
  )
}

predict_probabilities <- function(fit, test, recipe, design) {
  pars <- c(
    "beta",
    "z_country_intercept", "z_country_ntl", "z_country_pop",
    "sigma_country_intercept", "sigma_country_ntl", "sigma_country_pop",
    "z_stage_ntl", "z_stage_pop", "sigma_stage_ntl", "sigma_stage_pop",
    "z_family_intercept", "z_family_ntl", "z_family_pop",
    "z_survey_intercept",
    "sigma_family_intercept", "sigma_family_ntl", "sigma_family_pop",
    "sigma_survey_intercept"
  )
  draws <- rstan::extract(fit, pars = pars, permuted = TRUE)
  eta <- as.matrix(test[, ..offset_columns]) +
    design$X %*% posterior_mean(draws, "beta")

  z_country_intercept <- posterior_mean(draws, "z_country_intercept")
  z_country_ntl <- posterior_mean(draws, "z_country_ntl")
  z_country_pop <- posterior_mean(draws, "z_country_pop")
  z_stage_ntl <- posterior_mean(draws, "z_stage_ntl")
  z_stage_pop <- posterior_mean(draws, "z_stage_pop")
  z_family_intercept <- posterior_mean(draws, "z_family_intercept")
  z_family_ntl <- posterior_mean(draws, "z_family_ntl")
  z_family_pop <- posterior_mean(draws, "z_family_pop")
  z_survey_intercept <- posterior_mean(draws, "z_survey_intercept")

  sigma_country_intercept <- colMeans(draws$sigma_country_intercept)
  sigma_country_ntl <- colMeans(draws$sigma_country_ntl)
  sigma_country_pop <- colMeans(draws$sigma_country_pop)
  sigma_stage_ntl <- colMeans(draws$sigma_stage_ntl)
  sigma_stage_pop <- colMeans(draws$sigma_stage_pop)
  sigma_family_intercept <- colMeans(draws$sigma_family_intercept)
  sigma_family_ntl <- colMeans(draws$sigma_family_ntl)
  sigma_family_pop <- colMeans(draws$sigma_family_pop)
  sigma_survey_intercept <- colMeans(draws$sigma_survey_intercept)

  country_seen <- !is.na(design$country_id)
  stage_seen <- !is.na(design$stage_id)
  family_seen <- !is.na(design$family_id)
  survey_seen <- !is.na(design$survey_id)

  for (q in seq_along(nonreference_labels)) {
    if (any(country_seen)) {
      rows <- which(country_seen)
      ids <- design$country_id[rows]
      eta[rows, q] <- eta[rows, q] +
        sigma_country_intercept[q] * z_country_intercept[q, ids] +
        design$ntl_z[rows] * sigma_country_ntl[q] *
          z_country_ntl[q, ids] +
        design$pop_z[rows] * sigma_country_pop[q] *
          z_country_pop[q, ids]
    }
    if (any(stage_seen)) {
      rows <- which(stage_seen)
      ids <- design$stage_id[rows]
      eta[rows, q] <- eta[rows, q] +
        design$ntl_z[rows] * sigma_stage_ntl[q] * z_stage_ntl[q, ids] +
        design$pop_z[rows] * sigma_stage_pop[q] * z_stage_pop[q, ids]
    }
    if (any(family_seen)) {
      rows <- which(family_seen)
      ids <- design$family_id[rows]
      eta[rows, q] <- eta[rows, q] +
        sigma_family_intercept[q] * z_family_intercept[q, ids] +
        design$ntl_z[rows] * sigma_family_ntl[q] *
          z_family_ntl[q, ids] +
        design$pop_z[rows] * sigma_family_pop[q] *
          z_family_pop[q, ids]
    }
    if (any(survey_seen)) {
      rows <- which(survey_seen)
      ids <- design$survey_id[rows]
      eta[rows, q] <- eta[rows, q] +
        sigma_survey_intercept[q] * z_survey_intercept[q, ids]
    }
  }

  eta_full <- matrix(0, nrow(test), length(bmi_labels))
  eta_full[, bmi_labels != "bmi_20_25"] <- eta
  maximum <- apply(eta_full, 1L, max)
  probability <- exp(eta_full - maximum)
  probability <- probability / rowSums(probability)
  if (any(!is.finite(probability))) {
    stop("Non-finite posterior-mean prediction", call. = FALSE)
  }
  probability
}

metrics_from_prediction <- function(
    test, observed, predicted, baseline, w2, task
) {
  weight <- as.numeric(test$pseudo_n)
  obesity_columns <- match(
    c("bmi_30_35", "bmi_35_40", "bmi_ge_40"), bmi_labels
  )
  observed_obesity <- rowSums(observed[, obesity_columns, drop = FALSE])
  predicted_obesity <- rowSums(predicted[, obesity_columns, drop = FALSE])
  baseline_obesity <- rowSums(baseline[, obesity_columns, drop = FALSE])
  category_mae <- vapply(seq_along(bmi_labels), function(index) {
    100 * weighted_mean_safe(
      abs(observed[, index] - predicted[, index]), weight
    )
  }, numeric(1))
  baseline_category_mae <- vapply(seq_along(bmi_labels), function(index) {
    100 * weighted_mean_safe(
      abs(observed[, index] - baseline[, index]), weight
    )
  }, numeric(1))
  category_r2 <- vapply(seq_along(bmi_labels), function(index) {
    weighted_r2(observed[, index], predicted[, index], weight)
  }, numeric(1))
  names(category_mae) <- bmi_labels
  names(baseline_category_mae) <- bmi_labels
  names(category_r2) <- bmi_labels

  source_values <- sort(unique(test$source_family))
  source_w2 <- vapply(source_values, function(source_name) {
    rows <- test$source_family == source_name
    weighted_mean_safe(w2[rows], weight[rows])
  }, numeric(1))
  source_mae <- vapply(source_values, function(source_name) {
    rows <- test$source_family == source_name
    100 * weighted_mean_safe(
      rowMeans(abs(observed[rows, , drop = FALSE] -
        predicted[rows, , drop = FALSE])),
      weight[rows]
    )
  }, numeric(1))

  list(
    task_id = task$task_id,
    scheme = task$scheme,
    fold = task$fold,
    test_rows = nrow(test),
    test_countries = uniqueN(test$iso3),
    test_sources = uniqueN(test$source_family),
    effective_people = sum(weight),
    weighted_w2_bmi_kg_m2 = weighted_mean_safe(w2, weight),
    source_balanced_w2_bmi_kg_m2 = mean(source_w2),
    weighted_seven_category_mae_pp = 100 * weighted_mean_safe(
      rowMeans(abs(observed - predicted)), weight
    ),
    source_balanced_seven_category_mae_pp = mean(source_mae),
    baseline_weighted_seven_category_mae_pp = 100 * weighted_mean_safe(
      rowMeans(abs(observed - baseline)), weight
    ),
    weighted_obesity_mae_pp = 100 * weighted_mean_safe(
      abs(observed_obesity - predicted_obesity), weight
    ),
    baseline_weighted_obesity_mae_pp = 100 * weighted_mean_safe(
      abs(observed_obesity - baseline_obesity), weight
    ),
    weighted_obesity_r2 = weighted_r2(
      observed_obesity, predicted_obesity, weight
    ),
    within_anchor_obesity_deviation_r = weighted_correlation(
      observed_obesity - baseline_obesity,
      predicted_obesity - baseline_obesity,
      weight
    ),
    weighted_category_mae_pp = as.list(category_mae),
    baseline_weighted_category_mae_pp = as.list(baseline_category_mae),
    weighted_category_r2 = as.list(category_r2)
  )
}

write_prediction_outputs <- function(test, probability, task) {
  observed <- as.matrix(test[, ..count_columns]) / test$pseudo_n
  baseline <- close_composition(as.matrix(test[, ..bmi_labels]))
  w2 <- row_wasserstein2(observed, probability)
  obesity_columns <- match(
    c("bmi_30_35", "bmi_35_40", "bmi_ge_40"), bmi_labels
  )
  prediction <- test[, .(
    row_id_stage01, source, source_family, source_role, survey_id,
    iso3, country, geography_id, geography_level, year, sex, age_group,
    age_midpoint, development_stage, pseudo_n
  )]
  prediction[, `:=`(
    model = "regional_steps_augmented_bayesian_multinomial_logistic_normal_mrp",
    task_id = task$task_id,
    scheme = task$scheme,
    fold = task$fold,
    w2_bmi_kg_m2 = w2
  )]
  for (index in seq_along(bmi_labels)) {
    label <- bmi_labels[index]
    prediction[, paste0("observed_", label) := observed[, index]]
    prediction[, paste0("predicted_", label) := probability[, index]]
    prediction[, paste0("baseline_", label) := baseline[, index]]
  }
  prediction[, `:=`(
    observed_obesity = rowSums(observed[, obesity_columns, drop = FALSE]),
    predicted_obesity = rowSums(
      probability[, obesity_columns, drop = FALSE]
    ),
    baseline_obesity = rowSums(baseline[, obesity_columns, drop = FALSE])
  )]

  prediction_path <- file.path(
    stage_root, "validation", "predictions",
    paste0(task$task_id, "_predictions_v1.csv.gz")
  )
  fwrite(prediction, prediction_path)
  metrics <- metrics_from_prediction(
    test, observed, probability, baseline, w2, task
  )
  metric_path <- file.path(
    stage_root, "validation", "metrics",
    paste0(task$task_id, "_metrics_v1.json")
  )
  write_json_atomic(metrics, metric_path)

  source_metrics <- prediction[, .(
    rows = .N,
    effective_people = sum(pseudo_n),
    weighted_w2_bmi_kg_m2 = weighted_mean_safe(w2_bmi_kg_m2, pseudo_n),
    weighted_seven_category_mae_pp = 100 * weighted_mean_safe(
      rowMeans(abs(
        as.matrix(.SD[, paste0("observed_", bmi_labels), with = FALSE]) -
          as.matrix(.SD[, paste0("predicted_", bmi_labels), with = FALSE])
      )),
      pseudo_n
    ),
    weighted_obesity_mae_pp = 100 * weighted_mean_safe(
      abs(observed_obesity - predicted_obesity), pseudo_n
    )
  ), by = source_family]
  source_path <- file.path(
    stage_root, "validation", "by_source",
    paste0(task$task_id, "_by_source_v1.csv")
  )
  fwrite(source_metrics, source_path)

  list(
    prediction_path = prediction_path,
    metric_path = metric_path,
    source_path = source_path,
    metrics = metrics,
    closure_max_error = max(abs(rowSums(probability) - 1)),
    finite_predictions = all(is.finite(probability))
  )
}

fit_task <- function(task, data) {
  metadata_path <- file.path(
    stage_root, "models", "checkpoints",
    paste0(task$task_id, "_metadata_v1.json")
  )
  fit_path <- file.path(
    stage_root, "models", "checkpoints",
    paste0(task$task_id, "_fit_v1.rds")
  )
  recipe_path <- file.path(
    stage_root, "models", "checkpoints",
    paste0(task$task_id, "_recipe_v1.rds")
  )
  if (file.exists(metadata_path) || file.exists(fit_path) ||
      file.exists(recipe_path)) {
    stop("Stage 3 task output already exists: ", task$task_id, call. = FALSE)
  }

  train <- data[J(task$train_ids), nomatch = 0L]
  test <- if (length(task$test_ids)) {
    data[J(task$test_ids), nomatch = 0L]
  } else {
    data[0]
  }
  if (!nrow(train)) stop("No training rows for ", task$task_id, call. = FALSE)
  if (task$task_id != "full_model" && !nrow(test)) {
    stop("No test rows for ", task$task_id, call. = FALSE)
  }
  if (length(intersect(train$row_id_stage01, test$row_id_stage01))) {
    stop("Train-test leakage for ", task$task_id, call. = FALSE)
  }

  recipe <- build_recipe(train)
  train_design <- make_design(train, recipe)
  stan_data <- make_stan_data(train, recipe, train_design)
  saveRDS(
    list(
      task = task,
      feature_names = colnames(train_design$X),
      recipe = recipe,
      train_row_ids = train$row_id_stage01,
      test_row_ids = test$row_id_stage01
    ),
    recipe_path,
    compress = "xz"
  )

  settings <- if (task$task_id == "full_model") {
    list(
      iter = 15000L, output_samples = 1000L, tol_rel_obj = 0.003,
      adapt_iter = 200L, eval_elbo = 100L, eta = 0.1
    )
  } else if (task$task_id == "smoke_preflight") {
    list(
      iter = 1500L, output_samples = 100L, tol_rel_obj = 0.02,
      adapt_iter = 100L, eval_elbo = 100L, eta = 0.1
    )
  } else {
    list(
      iter = 9000L, output_samples = 200L, tol_rel_obj = 0.01,
      adapt_iter = 150L, eval_elbo = 100L, eta = 0.1
    )
  }

  metadata <- list(
    status = "started",
    task_id = task$task_id,
    scheme = task$scheme,
    fold = task$fold,
    train_rows = nrow(train),
    test_rows = nrow(test),
    excluded_rows = length(task$excluded_ids),
    train_countries = uniqueN(train$iso3),
    test_countries = uniqueN(test$iso3),
    train_sources = uniqueN(train$source_family),
    test_sources = uniqueN(test$source_family),
    P = ncol(train_design$X),
    features = colnames(train_design$X),
    seed = task_seed(task$task_id),
    inference = settings,
    started_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    R = R.version.string,
    rstan = as.character(packageVersion("rstan")),
    model_input = model_path,
    model_input_sha256 = digest(
      file = model_path, algo = "sha256", serialize = FALSE
    )
  )
  if (!is.null(task$train_cutoff_year)) {
    metadata$train_cutoff_year <- task$train_cutoff_year
    metadata$test_start_year <- task$test_start_year
    metadata$test_end_year <- task$test_end_year
    metadata$train_year_range <- range(train$year)
    metadata$test_year_range <- range(test$year)
  }
  write_json_atomic(metadata, metadata_path)

  model <- readRDS(model_path)
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(
    vb(
      model,
      data = stan_data,
      seed = task_seed(task$task_id),
      algorithm = "meanfield",
      iter = settings$iter,
      output_samples = settings$output_samples,
      tol_rel_obj = settings$tol_rel_obj,
      adapt_iter = settings$adapt_iter,
      eval_elbo = settings$eval_elbo,
      eta = settings$eta,
      init = 0
    ),
    warning = function(condition) {
      warnings <<- c(warnings, conditionMessage(condition))
      message("WARNING: ", conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - started
  saveRDS(fit, fit_path, compress = "gzip")

  output <- NULL
  if (nrow(test)) {
    test_design <- make_design(test, recipe)
    probability <- predict_probabilities(
      fit, test, recipe, test_design
    )
    output <- write_prediction_outputs(test, probability, task)
  }

  if (task$task_id == "full_model") {
    summary_parameters <- c(
      "beta",
      "sigma_country_intercept", "sigma_country_ntl", "sigma_country_pop",
      "sigma_stage_ntl", "sigma_stage_pop",
      "sigma_family_intercept", "sigma_family_ntl", "sigma_family_pop",
      "sigma_survey_intercept"
    )
    summary_matrix <- rstan::summary(
      fit, pars = summary_parameters,
      probs = c(0.025, 0.5, 0.975)
    )$summary
    summary_table <- as.data.table(summary_matrix, keep.rownames = "parameter")
    fwrite(
      summary_table,
      file.path(stage_root, "models", "full_model_parameter_summary_v1.csv")
    )
    fwrite(
      data.table(
        feature_index = seq_along(colnames(train_design$X)),
        feature = colnames(train_design$X)
      ),
      file.path(stage_root, "models", "full_model_feature_index_v1.csv")
    )
  }

  metadata$status <- "complete"
  metadata$elapsed_seconds <- elapsed
  metadata$completed_at_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
  metadata$warnings <- unique(warnings)
  metadata$warning_count <- length(unique(warnings))
  metadata$fit_path <- fit_path
  metadata$recipe_path <- recipe_path
  if (!is.null(output)) {
    metadata$prediction_path <- output$prediction_path
    metadata$metric_path <- output$metric_path
    metadata$finite_predictions <- output$finite_predictions
    metadata$prediction_closure_max_error <- output$closure_max_error
  }
  write_json_atomic(metadata, metadata_path)
  invisible(metadata)
}

flatten_metric <- function(metric) {
  flat <- data.table(
    task_id = metric$task_id,
    scheme = metric$scheme,
    fold = as.character(metric$fold),
    test_rows = metric$test_rows,
    test_countries = metric$test_countries,
    test_sources = metric$test_sources,
    effective_people = metric$effective_people,
    weighted_w2_bmi_kg_m2 = metric$weighted_w2_bmi_kg_m2,
    source_balanced_w2_bmi_kg_m2 = metric$source_balanced_w2_bmi_kg_m2,
    weighted_seven_category_mae_pp =
      metric$weighted_seven_category_mae_pp,
    source_balanced_seven_category_mae_pp =
      metric$source_balanced_seven_category_mae_pp,
    baseline_weighted_seven_category_mae_pp =
      metric$baseline_weighted_seven_category_mae_pp,
    weighted_obesity_mae_pp = metric$weighted_obesity_mae_pp,
    baseline_weighted_obesity_mae_pp =
      metric$baseline_weighted_obesity_mae_pp,
    weighted_obesity_r2 = metric$weighted_obesity_r2,
    within_anchor_obesity_deviation_r =
      metric$within_anchor_obesity_deviation_r
  )
  for (label in bmi_labels) {
    flat[, paste0("mae_pp_", label) :=
      as.numeric(metric$weighted_category_mae_pp[[label]])]
    flat[, paste0("r2_", label) :=
      as.numeric(metric$weighted_category_r2[[label]])]
  }
  flat
}

summarize_prediction_table <- function(prediction, scheme) {
  observed_columns <- paste0("observed_", bmi_labels)
  predicted_columns <- paste0("predicted_", bmi_labels)
  baseline_columns <- paste0("baseline_", bmi_labels)
  observed <- as.matrix(prediction[, ..observed_columns])
  predicted <- as.matrix(prediction[, ..predicted_columns])
  baseline <- as.matrix(prediction[, ..baseline_columns])
  weight <- prediction$pseudo_n
  w2 <- prediction$w2_bmi_kg_m2
  task <- list(task_id = scheme, scheme = scheme, fold = "pooled")
  metrics_from_prediction(
    prediction, observed, predicted, baseline, w2, task
  )
}

finalize_stage <- function() {
  internal_tasks <- paste0("internal_fold_", 1:5)
  country_tasks <- paste0("country_fold_", 1:5)
  validation_tasks <- c(internal_tasks, country_tasks)
  expected_tasks <- validation_tasks
  metadata_paths <- file.path(
    stage_root, "models", "checkpoints",
    paste0(expected_tasks, "_metadata_v1.json")
  )
  missing_metadata <- metadata_paths[!file.exists(metadata_paths)]
  if (length(missing_metadata)) {
    stop("Missing completed task metadata: ", missing_metadata[[1]], call. = FALSE)
  }
  metadata <- lapply(metadata_paths, read_json, simplifyVector = FALSE)
  incomplete <- vapply(metadata, function(value) {
    !identical(value$status, "complete")
  }, logical(1))
  if (any(incomplete)) {
    stop("Incomplete Stage 3 task: ", expected_tasks[which(incomplete)[1]], call. = FALSE)
  }

  metric_paths <- file.path(
    stage_root, "validation", "metrics",
    paste0(validation_tasks, "_metrics_v1.json")
  )
  task_metrics <- rbindlist(lapply(metric_paths, function(path) {
    flatten_metric(read_json(path, simplifyVector = FALSE))
  }), use.names = TRUE, fill = TRUE)
  fwrite(
    task_metrics,
    file.path(stage_root, "validation", "all_task_metrics_v1.csv")
  )

  prediction_paths <- file.path(
    stage_root, "validation", "predictions",
    paste0(validation_tasks, "_predictions_v1.csv.gz")
  )
  predictions <- rbindlist(
    lapply(prediction_paths, fread, showProgress = FALSE),
    use.names = TRUE,
    fill = TRUE
  )
  fwrite(
    predictions,
    file.path(stage_root, "validation", "all_validation_predictions_v1.csv.gz"),
    compress = "gzip"
  )
  internal_predictions <- predictions[
    scheme == "all_eligible20_internal_random_row_5fold"
  ]
  country_predictions <- predictions[
    scheme == "all_eligible20_country_grouped_5fold"
  ]
  scheme_summaries <- rbindlist(list(
    flatten_metric(summarize_prediction_table(
      internal_predictions, "all_eligible20_internal_random_row_5fold"
    )),
    flatten_metric(summarize_prediction_table(
      country_predictions, "all_eligible20_country_grouped_5fold"
    ))
  ), use.names = TRUE, fill = TRUE)
  fwrite(
    scheme_summaries,
    file.path(stage_root, "validation", "validation_scheme_summary_v1.csv")
  )
  fwrite(task_metrics[scheme ==
    "all_eligible20_internal_random_row_5fold"], file.path(
    stage_root, "validation", "internal_random_row_5fold_metrics_v1.csv"
  ))
  fwrite(task_metrics[scheme ==
    "all_eligible20_country_grouped_5fold"], file.path(
    stage_root, "validation", "country_grouped_5fold_metrics_v1.csv"
  ))
  fwrite(scheme_summaries[scheme ==
    "all_eligible20_internal_random_row_5fold"], file.path(
    stage_root, "validation", "internal_random_row_5fold_pooled_summary_v1.csv"
  ))
  fwrite(scheme_summaries[scheme ==
    "all_eligible20_country_grouped_5fold"], file.path(
    stage_root, "validation", "country_grouped_5fold_pooled_summary_v1.csv"
  ))

  predicted_columns <- paste0("predicted_", bmi_labels)
  closure_error <- max(abs(rowSums(
    as.matrix(predictions[, ..predicted_columns])
  ) - 1))
  finite_predictions <- all(is.finite(
    as.matrix(predictions[, ..predicted_columns])
  ))
  expected_validation_rows <- nrow(input$assignments[eligible_validation == TRUE])
  internal_unique <- !anyDuplicated(internal_predictions$row_id_stage01)
  country_unique <- !anyDuplicated(country_predictions$row_id_stage01)
  internal_countries <- sort(unique(internal_predictions$iso3))
  country_countries <- sort(unique(country_predictions$iso3))
  expected_countries <- sort(unique(input$assignments[
    eligible_validation == TRUE, iso3
  ]))
  country_leakage <- vapply(seq_len(5L), function(fold) {
    task <- resolve_task(
      paste0("country_fold_", fold), input$data, input$assignments
    )
    train_countries <- unique(input$data[J(task$train_ids), iso3])
    test_countries <- unique(input$data[J(task$test_ids), iso3])
    length(intersect(train_countries, test_countries)) > 0L
  }, logical(1))
  qa <- list(
    status = if (
      nrow(task_metrics) == 10L &&
      nrow(internal_predictions) == expected_validation_rows &&
      nrow(country_predictions) == expected_validation_rows &&
      internal_unique && country_unique &&
      identical(internal_countries, expected_countries) &&
      identical(country_countries, expected_countries) &&
      !any(country_leakage) &&
      !any(c("CHN", "JPN") %in% c(internal_countries, country_countries)) &&
      finite_predictions &&
      closure_error < 1e-10
    ) "passed" else "failed",
    stage = "all_eligible20_validation_v1",
    model = "regional_steps_augmented_bayesian_multinomial_logistic_normal_mrp",
    validation_schemes = c(
      "all_eligible20_internal_random_row_5fold",
      "all_eligible20_country_grouped_5fold"
    ),
    validation_tasks_complete = nrow(task_metrics),
    expected_validation_tasks = 10L,
    internal_validation_prediction_rows = nrow(internal_predictions),
    country_validation_prediction_rows = nrow(country_predictions),
    expected_prediction_rows_per_scheme = expected_validation_rows,
    internal_row_ids_unique = internal_unique,
    country_row_ids_unique = country_unique,
    country_holdout_leakage_by_fold = country_leakage,
    finite_predictions = finite_predictions,
    prediction_closure_max_error = closure_error,
    fixed_effect_features = 22L,
    internal_random_row_folds = 5L,
    country_holdout_folds = 5L,
    validation_countries = length(expected_countries),
    validation_country_codes = expected_countries,
    training_only_countries = training_only_countries,
    china_japan_in_test = any(c("CHN", "JPN") %in%
      c(internal_countries, country_countries)),
    national_only_steps_used_for_testing = FALSE,
    steps_regional_records_used_for_training = TRUE,
    steps_regional_records_used_for_validation = TRUE,
    manual_lambda_used = FALSE,
    inference = "rstan mean-field variational Bayes",
    grids_created = FALSE,
    random_forest_run = FALSE,
    tabpfn_run = FALSE
  )
  write_json_atomic(
    qa, file.path(stage_root, "qa", "all_eligible20_qa_summary_v1.json")
  )
  if (qa$status != "passed") stop("Unified validation QA failed", call. = FALSE)
}

input <- load_inputs()
if (task_id == "design_preflight") {
  preflight_task <- resolve_task(
    "country_fold_1", input$data, input$assignments
  )
  train <- input$data[J(preflight_task$train_ids), nomatch = 0L]
  test <- input$data[J(preflight_task$test_ids), nomatch = 0L]
  recipe <- build_recipe(train)
  train_design <- make_design(train, recipe)
  test_design <- make_design(test, recipe)
  preflight <- list(
    status = if (
      ncol(train_design$X) == 22L &&
      all(is.finite(train_design$X)) &&
      all(is.finite(test_design$X))
    ) "passed" else "failed",
    training_rows = nrow(train),
    test_rows = nrow(test),
    fixed_effect_features = ncol(train_design$X),
    feature_names = colnames(train_design$X),
    finite_training_design = all(is.finite(train_design$X)),
    finite_test_design = all(is.finite(test_design$X)),
    country_levels_training = length(recipe$country_levels),
    source_levels_training = length(recipe$family_levels),
    survey_levels_training = length(recipe$survey_levels),
    unseen_test_countries = sum(is.na(test_design$country_id)),
    unseen_test_sources = sum(is.na(test_design$family_id)),
    unseen_test_surveys = sum(is.na(test_design$survey_id))
  )
  write_json_atomic(
    preflight,
    file.path(stage_root, "qa", "all_eligible20_design_preflight_v1.json")
  )
  if (preflight$status != "passed") {
    stop("Stage 3 design preflight failed", call. = FALSE)
  }
} else if (task_id == "finalize") {
  finalize_stage()
} else {
  task <- resolve_task(task_id, input$data, input$assignments)
  fit_task(task, input$data)
}
