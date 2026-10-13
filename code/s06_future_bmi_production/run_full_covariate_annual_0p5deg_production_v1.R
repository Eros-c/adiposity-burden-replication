if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

Sys.setenv(BMI_ANNUAL_CORE_ONLY = "1")
core_path <- file.path(
  file.path(.cfg$RESULTS, "06_future_bmi_2025_2050"),
  "full_covariate_annual_0p5deg_production_v1_20260906/scripts",
  "full_covariate_annual_core_v1.R"
)
source(core_path, local = globalenv())

scenario_slugs <- c(
  "SSP1-2.6" = "ssp126", "SSP2-4.5" = "ssp245",
  "SSP3-7.0" = "ssp370", "SSP5-8.5" = "ssp585"
)

argument_value <- function(flag) {
  arguments <- commandArgs(trailingOnly = TRUE)
  index <- match(flag, arguments)
  if (is.na(index) || index == length(arguments)) return(NULL)
  arguments[index + 1L]
}

requested_scenario <- argument_value("--scenario")
requested_year <- argument_value("--year")
selected_scenarios <- if (is.null(requested_scenario)) {
  c("SSP2-4.5", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5")
} else strsplit(requested_scenario, ",", fixed = TRUE)[[1L]]
selected_years <- if (is.null(requested_year)) {
  production_years
} else as.integer(strsplit(requested_year, ",", fixed = TRUE)[[1L]])
if (any(!selected_scenarios %in% names(scenarios))) stop("Unknown --scenario")
if (any(!selected_years %in% production_years)) stop("--year must be within 2025-2050")

safe_band_name <- function(value) {
  value <- gsub("[+]", "plus", value)
  value <- gsub("-", "_", value, fixed = TRUE)
  value <- gsub("[^A-Za-z0-9_]", "_", value)
  make.unique(substr(value, 1L, 120L), sep = "_")
}

group_closure_error <- function(probability, strata) {
  errors <- numeric(strata)
  for (index in seq_len(strata)) {
    columns <- ((index - 1L) * length(bmi_labels) + 1L):(index * length(bmi_labels))
    row_total <- rowSums(probability[, columns, drop = FALSE])
    row_total <- row_total[is.finite(row_total)]
    errors[index] <- if (length(row_total)) max(abs(row_total - 1)) else 0
  }
  max(errors)
}

predict_country_stratified <- function(iso3, scenario, year, dynamic) {
  iso3_value <- iso3
  scenario_value <- scenario
  year_value <- year
  spatial <- build_spatial_input(iso3, scenario, year, dynamic)
  model_grid <- spatial$model_grid
  allocation <- spatial$allocation
  adult_population <- spatial$adult_population
  n_cells <- nrow(model_grid)
  n_categories <- length(bmi_labels)

  age_sex_probability <- matrix(
    NA_real_, n_cells, length(sexes) * length(age_groups) * n_categories
  )
  age_numerator <- matrix(0, n_cells, length(age_groups) * n_categories)
  sex_numerator <- matrix(0, n_cells, length(sexes) * n_categories)
  overall_numerator <- matrix(0, n_cells, n_categories)
  age_population <- matrix(0, n_cells, length(age_groups))
  sex_population <- matrix(0, n_cells, length(sexes))
  age_sex_names <- character(ncol(age_sex_probability))
  maximum_anchor_error <- 0
  maximum_raking_iterations <- 0L
  maximum_closure_error <- 0
  stratum_index <- 0L

  for (sex_index in seq_along(sexes)) {
    sex <- sexes[sex_index]
    for (age_index in seq_along(age_groups)) {
      age_group <- age_groups[age_index]
      stratum_index <- stratum_index + 1L
      demographic_population <- allocation[, paste(sex, age_group, sep = "|")]
      sex_value <- sex
      age_group_value <- age_group
      target_row <- anchor[J(
        scenario_value, iso3_value, year_value, sex_value, age_group_value
      )]
      if (nrow(target_row) != 1L) {
        stop("Missing BMI anchor: ", scenario, " ", iso3, " ", year, " ", sex, " ", age_group)
      }
      target <- setNames(
        as.numeric(unlist(target_row[, ..target_columns], use.names = FALSE)), bmi_labels
      )
      prediction_data <- copy(model_grid)
      prediction_data[, `:=`(
        sex = sex, age_group = age_group,
        model_age_midpoint = model_age_midpoint(age_group)
      )]
      probability <- rake_composition(
        predict_uncalibrated(prediction_data, target), target, demographic_population
      )$probability
      calibrated <- colSums(probability * demographic_population) /
        sum(demographic_population)
      maximum_anchor_error <- max(maximum_anchor_error, max(abs(calibrated - target)))
      maximum_closure_error <- max(
        maximum_closure_error, max(abs(rowSums(probability) - 1))
      )

      age_sex_columns <- ((stratum_index - 1L) * n_categories + 1L):
        (stratum_index * n_categories)
      age_columns <- ((age_index - 1L) * n_categories + 1L):(age_index * n_categories)
      sex_columns <- ((sex_index - 1L) * n_categories + 1L):(sex_index * n_categories)
      age_sex_probability[, age_sex_columns] <- probability
      age_sex_names[age_sex_columns] <- paste(sex, age_group, bmi_labels, sep = "__")
      age_numerator[, age_columns] <- age_numerator[, age_columns, drop = FALSE] +
        probability * demographic_population
      sex_numerator[, sex_columns] <- sex_numerator[, sex_columns, drop = FALSE] +
        probability * demographic_population
      overall_numerator <- overall_numerator + probability * demographic_population
      age_population[, age_index] <- age_population[, age_index] + demographic_population
      sex_population[, sex_index] <- sex_population[, sex_index] + demographic_population
    }
  }

  age_probability <- age_numerator
  for (index in seq_along(age_groups)) {
    columns <- ((index - 1L) * n_categories + 1L):(index * n_categories)
    age_probability[, columns] <- age_probability[, columns, drop = FALSE] /
      age_population[, index]
  }
  sex_probability <- sex_numerator
  for (index in seq_along(sexes)) {
    columns <- ((index - 1L) * n_categories + 1L):(index * n_categories)
    sex_probability[, columns] <- sex_probability[, columns, drop = FALSE] /
      sex_population[, index]
  }
  overall_probability <- overall_numerator / adult_population

  for (index in seq_len(ncol(allocation))) {
    columns <- ((index - 1L) * n_categories + 1L):(index * n_categories)
    age_sex_probability[allocation[, index] <= 0, columns] <- NA_real_
  }
  for (index in seq_len(ncol(age_population))) {
    columns <- ((index - 1L) * n_categories + 1L):(index * n_categories)
    age_probability[age_population[, index] <= 0, columns] <- NA_real_
  }
  for (index in seq_len(ncol(sex_population))) {
    columns <- ((index - 1L) * n_categories + 1L):(index * n_categories)
    sex_probability[sex_population[, index] <= 0, columns] <- NA_real_
  }

  colnames(age_sex_probability) <- age_sex_names
  colnames(age_probability) <- unlist(lapply(
    age_groups, function(age) paste(age, bmi_labels, sep = "__")
  ))
  colnames(sex_probability) <- unlist(lapply(
    sexes, function(sex) paste(sex, bmi_labels, sep = "__")
  ))
  colnames(overall_probability) <- bmi_labels
  colnames(age_population) <- age_groups
  colnames(sex_population) <- sexes
  colnames(allocation) <- paste(
    rep(sexes, each = length(age_groups)), rep(age_groups, times = length(sexes)),
    sep = "__"
  )

  qa <- data.table(
    scenario = scenario, iso3 = iso3, year = year,
    supported_cells = n_cells, allocated_adult_population = sum(adult_population),
    preliminary_population_retained_fraction = spatial$retained_fraction,
    maximum_anchor_error = maximum_anchor_error,
    maximum_raking_iterations = maximum_raking_iterations,
    maximum_probability_closure_error = max(
      maximum_closure_error,
      group_closure_error(age_probability, length(age_groups)),
      group_closure_error(sex_probability, length(sexes)),
      max(abs(rowSums(overall_probability) - 1))
    ),
    maximum_centering_error = spatial$maximum_centering_error,
    imputed_covariate_values = spatial$imputed_covariate_values,
    population_fallback_cells = spatial$population_fallback_cells,
    country_effect_applied = iso3 %in% recipe$country_levels,
    stage_effect_applied = !is.na(model_grid$development_stage[1]) &&
      model_grid$development_stage[1] %in% recipe$stage_levels
  )
  list(
    cells = model_grid$cell_id,
    probability = list(
      overall = overall_probability, sex = sex_probability,
      age = age_probability, age_sex = age_sex_probability
    ),
    population = list(
      overall = matrix(adult_population, ncol = 1L,
        dimnames = list(NULL, "adult_population")),
      sex = sex_population, age = age_population, age_sex = allocation
    ),
    qa = qa, extrapolation = spatial$extrapolation
  )
}

validate_cog <- function(path, layers) {
  if (!file.exists(path) || file.info(path)$size <= 0) return(FALSE)
  candidate <- try(rast(path), silent = TRUE)
  if (inherits(candidate, "try-error")) return(FALSE)
  valid <- nlyr(candidate) == layers &&
    isTRUE(compareGeom(candidate, country_raster, stopOnError = FALSE,
      crs = TRUE, ext = TRUE, rowcol = TRUE, res = TRUE))
  rm(candidate)
  valid
}

write_cog <- function(sparse_values, cells, band_names, path, quantity) {
  sparse_values <- as.matrix(sparse_values)
  if (nrow(sparse_values) != length(cells) || any(is.infinite(sparse_values))) {
    stop("Invalid sparse output matrix: ", path)
  }
  if (quantity == "population" && any(!is.finite(sparse_values))) {
    stop("Non-finite population output: ", path)
  }
  finite_values <- sparse_values[is.finite(sparse_values)]
  if (!length(finite_values)) stop("Output contains no finite values: ", path)
  if (quantity == "probability" &&
      (min(finite_values) < -1e-10 || max(finite_values) > 1 + 1e-10)) {
    stop("Probability outside [0,1]: ", path)
  }
  if (quantity == "population" && min(finite_values) < -1e-8) {
    stop("Negative population: ", path)
  }
  if (!file.exists(path)) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    partial <- paste0(path, ".partial.tif")
    if (file.exists(partial)) unlink(partial)
    full_values <- matrix(NA_real_, ncell(country_raster), ncol(sparse_values))
    full_values[cells, ] <- sparse_values
    output <- rast(
      nrows = nrow(country_raster), ncols = ncol(country_raster),
      nlyrs = ncol(sparse_values), xmin = xmin(country_raster),
      xmax = xmax(country_raster), ymin = ymin(country_raster),
      ymax = ymax(country_raster), crs = crs(country_raster)
    )
    values(output) <- full_values
    names(output) <- safe_band_name(band_names)
    writeRaster(
      output, partial, overwrite = TRUE, filetype = "COG", datatype = "FLT4S",
      NAflag = -9999,
      gdal = c("COMPRESS=ZSTD", "LEVEL=9", "BLOCKSIZE=512", "BIGTIFF=IF_SAFER")
    )
    rm(full_values, output)
    gc()
    if (!validate_cog(partial, ncol(sparse_values))) stop("Invalid new COG: ", partial)
    if (!file.rename(partial, path)) stop("Could not finalize COG: ", path)
  }
  if (!validate_cog(path, ncol(sparse_values))) stop("Invalid existing COG: ", path)
  data.table(
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    quantity = quantity, layers = ncol(sparse_values),
    sparse_min = min(finite_values), sparse_max = max(finite_values),
    sparse_nodata_values = sum(!is.finite(sparse_values)),
    bytes = file.info(path)$size, sha256 = sha256_file(path)
  )
}

task_paths <- function(scenario, year) {
  directory <- file.path(root, "outputs", scenario_slugs[[scenario]], year)
  groups <- c("overall", "sex", "age", "age_sex")
  list(
    probability = setNames(file.path(directory, paste0(
      "bmi7_probability_", groups, "_0p5deg_v1.tif"
    )), groups),
    population = setNames(file.path(directory, paste0(
      "adult_population_", groups, "_0p5deg_v1.tif"
    )), groups)
  )
}

band_map_path <- file.path(root, "contracts", "annual_0p5deg_band_map_v1.csv")
nodata_amendment_path <- file.path(
  root, "contracts", "annual_0p5deg_zero_population_nodata_amendment_v1.json"
)
if (!file.exists(nodata_amendment_path)) write_new_json(list(
  status = "FROZEN_AFTER_SMOKE_TEST_BEFORE_PRODUCTION",
  reason = paste(
    "A stratum can have zero allocated population in an otherwise retained cell;",
    "its BMI probability is undefined and is stored as NoData rather than imputed."
  ),
  probability_nodata = -9999,
  population_requirement = "All population bands remain finite and non-negative.",
  overall_probability_requirement = "Overall probabilities remain finite and closed.",
  stratified_probability_requirement = paste(
    "Finite values must be in [0,1] and close to one within each stratum;",
    "NoData is allowed only where the companion stratum population is zero."
  )
), nodata_amendment_path)
if (!file.exists(band_map_path)) {
  probability_names <- list(
    overall = bmi_labels,
    sex = unlist(lapply(sexes, function(sex) paste(sex, bmi_labels, sep = "__"))),
    age = unlist(lapply(age_groups, function(age) paste(age, bmi_labels, sep = "__"))),
    age_sex = unlist(lapply(sexes, function(sex) unlist(lapply(
      age_groups, function(age) paste(sex, age, bmi_labels, sep = "__")
    ))))
  )
  population_names <- list(
    overall = "adult_population", sex = sexes, age = age_groups,
    age_sex = unlist(lapply(sexes, function(sex) paste(sex, age_groups, sep = "__")))
  )
  maps <- list()
  map_index <- 0L
  for (group in names(probability_names)) {
    map_index <- map_index + 1L
    maps[[map_index]] <- data.table(
      file_type = "probability", group = group,
      band = seq_along(probability_names[[group]]),
      band_name = safe_band_name(probability_names[[group]]), unit = "proportion_0_to_1"
    )
    map_index <- map_index + 1L
    maps[[map_index]] <- data.table(
      file_type = "population", group = group,
      band = seq_along(population_names[[group]]),
      band_name = safe_band_name(population_names[[group]]), unit = "persons_age25plus"
    )
  }
  write_new_csv(rbindlist(maps), band_map_path, FALSE)
}

run_task <- function(scenario, year, task_number, task_count) {
  task_id <- paste0(scenario_slugs[[scenario]], "_", year)
  done_path <- file.path(root, "checkpoints", paste0(task_id, ".done.json"))
  if (file.exists(done_path)) {
    done <- read_json(done_path, simplifyVector = TRUE)
    expected_outputs <- task_paths(scenario, year)
    expected_outputs <- c(unname(expected_outputs$probability), unname(expected_outputs$population))
    if (!all(file.exists(expected_outputs))) stop("Incomplete checkpoint: ", task_id)
    message(sprintf("[%d/%d] SKIP %s", task_number, task_count, task_id))
    return(invisible(done))
  }

  message(sprintf("[%d/%d] START %s", task_number, task_count, task_id))
  task_started <- timestamp_utc()
  dynamic <- load_dynamic(scenario, year)
  results <- vector("list", length(spatial_countries))
  for (index in seq_along(spatial_countries)) {
    if (index %% 25L == 0L || index == length(spatial_countries)) {
      message(sprintf("  %s country %d/%d", task_id, index, length(spatial_countries)))
    }
    results[[index]] <- predict_country_stratified(
      spatial_countries[index], scenario, year, dynamic
    )
  }
  rm(dynamic)
  gc()

  cells <- unlist(lapply(results, `[[`, "cells"), use.names = FALSE)
  if (anyDuplicated(cells)) stop("Duplicate grid cells in ", task_id)
  country_qa <- rbindlist(lapply(results, `[[`, "qa"))
  extrapolation <- rbindlist(lapply(results, `[[`, "extrapolation"))
  groups <- c("overall", "sex", "age", "age_sex")
  probability <- setNames(lapply(groups, function(group) do.call(
    rbind, lapply(results, function(value) value$probability[[group]])
  )), groups)
  population <- setNames(lapply(groups, function(group) do.call(
    rbind, lapply(results, function(value) value$population[[group]])
  )), groups)
  rm(results)
  gc()

  paths <- task_paths(scenario, year)
  cog_qa <- list()
  cog_index <- 0L
  for (group in groups) {
    population_by_probability_band <- population[[group]][,
      rep(seq_len(ncol(population[[group]])), each = length(bmi_labels)),
      drop = FALSE
    ]
    invalid_nodata <- !is.finite(probability[[group]]) &
      population_by_probability_band > 1e-12
    if (any(invalid_nodata)) {
      stop("Probability NoData found where stratum population is positive: ", task_id, " ", group)
    }
    cog_index <- cog_index + 1L
    cog_qa[[cog_index]] <- write_cog(
      probability[[group]], cells, colnames(probability[[group]]),
      paths$probability[[group]], "probability"
    )[, `:=`(scenario = scenario, year = year, group = group)]
    cog_index <- cog_index + 1L
    cog_qa[[cog_index]] <- write_cog(
      population[[group]], cells, colnames(population[[group]]),
      paths$population[[group]], "population"
    )[, `:=`(scenario = scenario, year = year, group = group)]
  }
  cog_qa <- rbindlist(cog_qa)
  total_population <- sum(population$overall[, 1])
  aggregate_probability <- colSums(
    probability$overall * population$overall[, 1] / total_population
  )
  aggregate <- data.table(
    scenario = scenario, year = year,
    population_basis = "scenario_specific_spatial_177_country_population",
    spatial_countries = length(spatial_countries), supported_cells = length(cells),
    adult_population = total_population
  )
  for (index in seq_along(bmi_labels)) {
    aggregate[, paste0("p_", bmi_labels[index]) := aggregate_probability[index]]
  }
  aggregate[, `:=`(
    obesity = sum(aggregate_probability[5:7]),
    mean_bmi = sum(aggregate_probability * support)
  )]
  task_qa <- data.table(
    scenario = scenario, year = year, task_id = task_id,
    countries = nrow(country_qa), supported_cells = length(cells),
    maximum_anchor_error_pp = 100 * max(country_qa$maximum_anchor_error),
    maximum_probability_closure_error = max(country_qa$maximum_probability_closure_error),
    maximum_centering_error = max(country_qa$maximum_centering_error),
    minimum_population_retained_fraction = min(
      country_qa$preliminary_population_retained_fraction
    ),
    population_fallback_cells = sum(country_qa$population_fallback_cells),
    imputed_covariate_values = sum(country_qa$imputed_covariate_values),
    cog_files = nrow(cog_qa), cog_bytes = sum(cog_qa$bytes)
  )

  qa_directory <- file.path(root, "qa", "tasks", scenario_slugs[[scenario]], year)
  dir.create(qa_directory, recursive = TRUE, showWarnings = FALSE)
  qa_files <- c(
    country = file.path(qa_directory, paste0(task_id, "_country_QA_v1.csv")),
    extrapolation = file.path(qa_directory, paste0(task_id, "_extrapolation_QA_v1.csv.gz")),
    cog = file.path(qa_directory, paste0(task_id, "_COG_QA_v1.csv")),
    task = file.path(qa_directory, paste0(task_id, "_task_QA_v1.csv")),
    aggregate = file.path(qa_directory, paste0(task_id, "_aggregate_v1.csv"))
  )
  if (!file.exists(qa_files[["country"]])) write_new_csv(country_qa, qa_files[["country"]], FALSE)
  if (!file.exists(qa_files[["extrapolation"]])) write_new_csv(extrapolation, qa_files[["extrapolation"]])
  if (!file.exists(qa_files[["cog"]])) write_new_csv(cog_qa, qa_files[["cog"]], FALSE)
  if (!file.exists(qa_files[["task"]])) write_new_csv(task_qa, qa_files[["task"]], FALSE)
  if (!file.exists(qa_files[["aggregate"]])) write_new_csv(aggregate, qa_files[["aggregate"]], FALSE)

  output_files <- c(unname(paths$probability), unname(paths$population))
  done <- list(
    task_id = task_id, scenario = scenario, year = year, status = "PASS_WITH_WARN",
    started_at_utc = task_started, completed_at_utc = timestamp_utc(),
    output_files = normalizePath(output_files, winslash = "/", mustWork = TRUE),
    qa_files = normalizePath(qa_files, winslash = "/", mustWork = TRUE),
    warning = "Future covariate extrapolation remains reportable"
  )
  write_new_json(done, done_path)
  rm(probability, population)
  gc()
  message(sprintf("[%d/%d] DONE %s", task_number, task_count, task_id))
  invisible(done)
}

started_at <- timestamp_utc()
selected_tasks <- CJ(scenario = selected_scenarios, year = selected_years, unique = TRUE)
for (index in seq_len(nrow(selected_tasks))) {
  run_task(selected_tasks$scenario[index], selected_tasks$year[index], index, nrow(selected_tasks))
}

task_qa_paths <- list.files(
  file.path(root, "qa", "tasks"), pattern = "_task_QA_v1[.]csv$",
  recursive = TRUE, full.names = TRUE
)
task_qa <- if (length(task_qa_paths)) rbindlist(lapply(task_qa_paths, fread)) else data.table()
expected_tasks <- length(scenarios) * length(production_years)
completed_tasks <- length(list.files(
  file.path(root, "checkpoints"), pattern = "[.]done[.]json$", full.names = TRUE
))
progress <- list(
  status = if (completed_tasks == expected_tasks) "COMPLETE_PENDING_FINAL_QA" else "IN_PROGRESS",
  updated_at_utc = timestamp_utc(), completed_tasks = completed_tasks,
  expected_tasks = expected_tasks, completed_fraction = completed_tasks / expected_tasks
)
progress_path <- file.path(root, "qa", "annual_0p5deg_run_progress_v1.json")
write_json(progress, progress_path, pretty = TRUE, auto_unbox = TRUE, na = "null")

if (completed_tasks == expected_tasks) {
  setorder(task_qa, scenario, year)
  task_qa_output <- file.path(root, "qa", "annual_0p5deg_task_QA_v1.csv")
  if (!file.exists(task_qa_output)) write_new_csv(task_qa, task_qa_output, FALSE)
  aggregate_paths <- list.files(
    file.path(root, "qa", "tasks"), pattern = "_aggregate_v1[.]csv$",
    recursive = TRUE, full.names = TRUE
  )
  aggregate <- rbindlist(lapply(aggregate_paths, fread))
  setorder(aggregate, scenario, year)
  aggregate_output <- file.path(root, "data", "annual_0p5deg_spatial_aggregate_v1.csv")
  if (!file.exists(aggregate_output)) write_new_csv(aggregate, aggregate_output, FALSE)

  common_2025_error <- 0
  reference <- task_paths("SSP2-4.5", 2025L)
  for (scenario in setdiff(names(scenarios), "SSP2-4.5")) {
    alternative <- task_paths(scenario, 2025L)
    for (group in names(reference$probability)) {
      difference <- global(
        abs(rast(reference$probability[[group]]) - rast(alternative$probability[[group]])),
        "max", na.rm = TRUE
      )[[1L]]
      common_2025_error <- max(common_2025_error, difference)
    }
  }
  gate <- data.table(
    check_id = c(
      "R01_TASK_COMPLETENESS", "R02_NATIONAL_ANCHOR", "R03_COMPOSITION_CLOSURE",
      "R04_WITHIN_COUNTRY_CENTERING", "R05_POPULATION_MASK_RETENTION",
      "R06_COG_COMPLETENESS", "R07_COMMON_2025_SURFACE"
    ),
    observed = c(
      nrow(task_qa), max(task_qa$maximum_anchor_error_pp),
      max(task_qa$maximum_probability_closure_error),
      max(task_qa$maximum_centering_error),
      min(task_qa$minimum_population_retained_fraction),
      sum(task_qa$cog_files), common_2025_error
    ),
    threshold = c(
      expected_tasks, limits$maximum_national_anchor_error_pp,
      limits$maximum_probability_closure_error, limits$maximum_centering_error,
      limits$minimum_country_scenario_year_population_retained,
      expected_tasks * 8L, 1e-6
    ),
    direction = c(
      "equal", "less_or_equal", "less_or_equal", "less_or_equal",
      "greater_or_equal", "equal", "less_or_equal"
    )
  )
  gate[, status := fifelse(
    (direction == "equal" & observed == threshold) |
      (direction == "less_or_equal" & observed <= threshold) |
      (direction == "greater_or_equal" & observed >= threshold), "PASS", "FAIL"
  )]
  overall_status <- if (all(gate$status == "PASS")) "PASS_WITH_WARN" else "FAIL"
  gate_path <- file.path(root, "qa", "annual_0p5deg_production_gate_v1.csv")
  if (!file.exists(gate_path)) write_new_csv(gate, gate_path, FALSE)
  summary <- list(
    status = overall_status, started_at_utc = started_at,
    completed_at_utc = timestamp_utc(), spatial_countries = length(spatial_countries),
    national_only_locations = nrow(disposition[disposition == "NATIONAL_ONLY_NO_0P5_CELL"]),
    scenarios = names(scenarios), years = production_years,
    tasks = nrow(task_qa), cog_files = sum(task_qa$cog_files),
    output_gib = sum(task_qa$cog_bytes) / 1024^3,
    maximum_anchor_error_pp = max(task_qa$maximum_anchor_error_pp),
    maximum_probability_closure_error = max(task_qa$maximum_probability_closure_error),
    minimum_population_retained_fraction = min(task_qa$minimum_population_retained_fraction),
    disease_burden_analysis_authorized = FALSE
  )
  summary_path <- file.path(root, "qa", "annual_0p5deg_production_summary_v1.json")
  if (!file.exists(summary_path)) write_new_json(summary, summary_path)
  report <- c(
    "# Annual 2025-2050 full-covariate BMI production at 0.5 degree",
    "", paste0("Status: **", overall_status, "**"), "", "## Scope", "",
    "The validated 22-feature Bayesian multinomial logistic-normal MRP was applied annually under four SSP scenarios.",
    "The 177 spatial locations have overall, sex, age, and age-by-sex seven-category probability COGs and matching population COGs.",
    "The 27 locations without a valid 0.5-degree country cell remain available in the national anchor table only.",
    "", "## Quality gates", "", paste(capture.output(print(gate)), collapse = "\n"),
    "", "## Warnings", "",
    "Future covariate extrapolation remains reportable.",
    "COK and GRL contain SSP population-grid gaps handled using calibrated adult-population fallback.",
    "Disease-burden analysis remains unauthorized until 0.1-degree nesting and ADM1 aggregation pass their own gates."
  )
  report_path <- file.path(root, "reports", "annual_0p5deg_production_report_v1.md")
  if (!file.exists(report_path)) write_new_text(report, report_path)
  cat(toJSON(summary, pretty = TRUE, auto_unbox = TRUE), "\n")
} else {
  cat(toJSON(progress, pretty = TRUE, auto_unbox = TRUE), "\n")
}
