if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

try(Sys.setlocale("LC_CTYPE", "Chinese"), silent = TRUE)
suppressPackageStartupMessages({
  library(terra)
  library(data.table)
})

hist_root <- file.path(.cfg$ROOT, "future_scenario_ascii", "05_historical_bmi_1990_2024", "historical_25plus_compat_v1_20260909")
future_root <- file.path(.cfg$ROOT, "future_scenario_ascii", "06_future_bmi_2025_2050", "full_covariate_annual_0p1deg_production_v1_20260906")
zone_path <- file.path(.cfg$ROOT, "future_scenario_ascii", "06_future_bmi_2025_2050", "phase6i_remaining_scenarios_0p1deg_adm1_v1_20260904", "metadata", "adm1_zone_country_constrained_0p1deg_v3.tif")
lookup_path <- file.path(.cfg$ROOT, "future_scenario_ascii", "06_future_bmi_2025_2050", "phase6i_remaining_scenarios_0p1deg_adm1_v1_20260904", "metadata", "adm1_zone_lookup_v1.csv")
out_root <- file.path(hist_root, "qa")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

hist_file_labels <- c("lt18p5", "18p5_to_lt20", "20_to_lt25", "25_to_lt30", "30_to_lt35", "35_to_lt40", "ge40")
common_labels <- c("BMI <18.5", "BMI 18.5-<20", "BMI 20-<25", "BMI 25-<30", "BMI 30-<35", "BMI 35-<40", "BMI >=40")

hist_pop <- rast(file.path(hist_root, "data", "0p1deg", "2024", "global_2024_adult_population_regional_steps_mrp_0p1deg.tif"))[[1]]
hist_probability_paths <- vapply(hist_file_labels, function(x) file.path(hist_root, "data", "0p1deg", "2024", sprintf("global_2024_bmi_%s_regional_steps_mrp_0p1deg_bp.tif", x)), character(1))
hist_prob <- Reduce(function(left, right) c(left, right), lapply(hist_probability_paths, function(path) rast(path)[[1]]))
hist_prob <- hist_prob / 10000
names(hist_prob) <- common_labels

future_pop <- rast(file.path(future_root, "outputs", "ssp245", "2025", "population_overall_ssp245_2025_0p1deg_v1.tif"))
future_prob <- rast(file.path(future_root, "outputs", "ssp245", "2025", "probability_overall_ssp245_2025_0p1deg_v1.tif"))
names(future_prob) <- common_labels

zone <- crop(rast(zone_path), hist_pop)
future_pop <- crop(future_pop, hist_pop)
future_prob <- crop(future_prob, hist_pop)
if (!compareGeom(hist_pop, future_pop, zone, stopOnError = FALSE)) stop("Historical, future and zone geometries do not align")

weighted_by_country <- function(probability, population, zones) {
  valid_population <- ifel(is.finite(population) & population > 0, population, NA)
  denominator <- as.data.table(zonal(valid_population, zones, fun = "sum", na.rm = TRUE))
  setnames(denominator, c("zone", "population"))
  numerator <- lapply(seq_len(nlyr(probability)), function(i) {
    x <- probability[[i]] * valid_population
    y <- as.data.table(zonal(x, zones, fun = "sum", na.rm = TRUE))
    setnames(y, c("zone", names(probability)[i]))
    y
  })
  out <- Reduce(function(left, right) merge(left, right, by = "zone", all = TRUE), c(list(denominator), numerator))
  for (x in names(probability)) out[, (x) := get(x) / population]
  out
}

lookup <- fread(lookup_path)[, .(zone = zone_id, iso3 = adm0_a3)]
hist_zone <- merge(weighted_by_country(hist_prob, hist_pop, zone), lookup, by = "zone", all.x = TRUE)
future_zone <- merge(weighted_by_country(future_prob, future_pop, zone), lookup, by = "zone", all.x = TRUE)

aggregate_country <- function(x) {
  x <- x[is.finite(population) & population > 0 & !is.na(iso3)]
  out <- x[, .(adult_population = sum(population)), by = iso3]
  for (metric in common_labels) {
    numerator <- x[, .(numerator = sum(get(metric) * population)), by = iso3]
    out <- merge(out, numerator, by = "iso3", all.x = TRUE, sort = FALSE)
    out[, (metric) := numerator / adult_population]
    out[, numerator := NULL]
  }
  out[, obesity := `BMI 30-<35` + `BMI 35-<40` + `BMI >=40`]
  out[, overweight_obesity := `BMI 25-<30` + obesity]
  out[, mean_bmi_midpoint := as.matrix(.SD) %*% c(17, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5), .SDcols = common_labels]
  out
}

hist_country <- aggregate_country(hist_zone)
future_country <- aggregate_country(future_zone)
setnames(hist_country, setdiff(names(hist_country), "iso3"), paste0(setdiff(names(hist_country), "iso3"), "_2024"))
setnames(future_country, setdiff(names(future_country), "iso3"), paste0(setdiff(names(future_country), "iso3"), "_2025"))
country <- merge(hist_country, future_country, by = "iso3", all = TRUE)
for (metric in c(common_labels, "obesity", "overweight_obesity", "mean_bmi_midpoint")) {
  country[, (paste0(metric, "_delta_2025_minus_2024")) := get(paste0(metric, "_2025")) - get(paste0(metric, "_2024"))]
}

fwrite(country, file.path(out_root, "historical2024_ssp245_2025_country_continuity_v1.csv"), bom = TRUE)

global_summary <- rbindlist(list(
  data.table(period = "Historical MRP 2024", weighted_by_country = "historical_population", t(colSums(hist_zone[, ..common_labels] * hist_zone$population, na.rm = TRUE) / sum(hist_zone$population, na.rm = TRUE))),
  data.table(period = "Future SSP2-4.5 2025", weighted_by_country = "future_population", t(colSums(future_zone[, ..common_labels] * future_zone$population, na.rm = TRUE) / sum(future_zone$population, na.rm = TRUE)))
), fill = TRUE)
for (table_row in seq_len(nrow(global_summary))) {
  global_summary[table_row, obesity := `BMI 30-<35` + `BMI 35-<40` + `BMI >=40`]
  global_summary[table_row, overweight_obesity := `BMI 25-<30` + obesity]
  global_summary[table_row, mean_bmi_midpoint := as.numeric(as.matrix(.SD) %*% c(17, 19.25, 22.5, 27.5, 32.5, 37.5, 42.5)), .SDcols = common_labels]
}
global_summary[, `:=`(obesity = obesity, overweight_obesity = overweight_obesity, mean_bmi_midpoint = mean_bmi_midpoint)]
fwrite(global_summary, file.path(out_root, "historical2024_ssp245_2025_global_continuity_v1.csv"), bom = TRUE)

hist_values <- values(hist_prob, mat = TRUE)
future_values <- values(future_prob, mat = TRUE)
hist_closure <- max(abs(rowSums(hist_values, na.rm = TRUE)[values(hist_pop, mat = FALSE) > 0] - 1), na.rm = TRUE)
future_closure <- max(abs(rowSums(future_values, na.rm = TRUE)[values(future_pop, mat = FALSE) > 0] - 1), na.rm = TRUE)
delta_cols <- grep("_delta_2025_minus_2024$", names(country), value = TRUE)
summary_lines <- c(
  "# Historical 2024 to SSP2-4.5 2025 continuity check",
  "",
  "This is a transition diagnostic, not an external validation. The historical product is anchored to NCD-RisC; the 2025 product is anchored to GBD.",
  "",
  sprintf("Historical supported-pixel composition closure maximum absolute error: %.3g", hist_closure),
  sprintf("Future supported-pixel composition closure maximum absolute error: %.3g", future_closure),
  sprintf("Countries with both products: %d", sum(is.finite(country$adult_population_2024) & is.finite(country$adult_population_2025))),
  "",
  "Global composition and derived metric values are in the accompanying CSV. Country-level deltas are 2025 minus 2024, with prevalence expressed as proportions and mean BMI as kg/m2.",
  "",
  "No pass/fail threshold is assigned to exposure change because a one-year biological change and the change of national anchor are not separately identifiable in this diagnostic."
)
writeLines(summary_lines, file.path(out_root, "historical2024_ssp245_2025_continuity_summary_v1.md"), useBytes = TRUE)
if (!is.finite(hist_closure) || hist_closure > 1e-4 || !is.finite(future_closure) || future_closure > 1e-4) stop("Composition closure check failed")
if (any(country[, rowSums(.SD) < 0.9999 | rowSums(.SD) > 1.0001, .SDcols = paste0(common_labels, "_2024")], na.rm = TRUE) ||
    any(country[, rowSums(.SD) < 0.9999 | rowSums(.SD) > 1.0001, .SDcols = paste0(common_labels, "_2025")], na.rm = TRUE)) stop("Country composition closure check failed")
cat("CONTINUITY_QA_PASS\n")
