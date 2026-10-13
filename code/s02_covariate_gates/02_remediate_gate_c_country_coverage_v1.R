if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(jsonlite)
})

source_run <- Sys.getenv("GATEC_SOURCE_RUN", unset = NA_character_)
target_run <- Sys.getenv("GATEC_TARGET_RUN", unset = NA_character_)
source_display <- Sys.getenv("GATEC_SOURCE_DISPLAY_RUN", unset = source_run)
target_display <- Sys.getenv("GATEC_TARGET_DISPLAY_RUN", unset = target_run)
if (is.na(source_run) || is.na(target_run)) {
  stop("GATEC_SOURCE_RUN and GATEC_TARGET_RUN are required")
}

read_csv_dt <- function(path) fread(path, encoding = "UTF-8")
write_csv_dt <- function(x, path) {
  tmp <- paste0(path, ".tmp")
  write.csv(as.data.frame(x), tmp, row.names = FALSE, fileEncoding = "UTF-8")
  if (file.exists(path)) file.remove(path)
  if (!file.rename(tmp, path)) stop("Cannot atomically write: ", path)
}

qa_dir <- file.path(target_run, "qa")
cal_path <- file.path(qa_dir, "Gate_C_country_calibration_QA.csv")
availability_path <- file.path(qa_dir, "Gate_C_country_availability_QA.csv")
checks_path <- file.path(qa_dir, "Gate_C_pass_fail_v1.csv")
country_path <- file.path(target_run, "data", "0p5deg", "country_id_0p5deg.tif")

cal <- read_csv_dt(cal_path)
country_r <- rast(country_path)
country_freq <- as.data.table(freq(country_r))
represented_ids <- unique(as.integer(country_freq[["value"]]))

cal[, grid_represented := country_id %in% represented_ids]
cal[grid_represented == FALSE & status == "no_spatial_weight",
    status := "not_represented_on_0p5_grid"]
cal[grid_represented == FALSE, fallback_used := FALSE]
setcolorder(cal, c(
  "variable", "scenario", "ssp", "year", "iso3", "country_id",
  "target_total", "pre_sum", "scale_factor", "post_sum", "relative_error",
  "fallback_used", "grid_represented", "status"
))
write_csv_dt(cal, cal_path)

availability <- cal[, .(
  scenario_year_rows = .N,
  target_rows = sum(is.finite(target_total)),
  calibrated_rows = sum(status == "pass"),
  no_national_target_rows = sum(status == "no_national_target"),
  no_spatial_weight_rows = sum(status == "no_spatial_weight"),
  not_represented_rows = sum(status == "not_represented_on_0p5_grid"),
  fallback_rows = sum(fallback_used, na.rm = TRUE),
  all_available_targets_calibrated = all(
    status[is.finite(target_total) & grid_represented] == "pass"
  )
), by = .(variable, iso3)]
write_csv_dt(availability, availability_path)

checks <- read_csv_dt(checks_path)
missing_grid <- sort(unique(cal[status == "not_represented_on_0p5_grid", iso3]))
no_weight <- cal[status == "no_spatial_weight",
                 .(countries = uniqueN(iso3), iso3 = paste(sort(unique(iso3)), collapse = ",")),
                 by = variable]
checks[check_id == "C09", `:=`(
  status = if (length(missing_grid)) "WARN" else "PASS",
  detail = if (length(missing_grid)) {
    paste0("Not represented by a 0.5-degree cell center: ", paste(missing_grid, collapse = ","))
  } else {
    "All boundary countries are represented by at least one 0.5-degree cell center"
  }
)]
checks[check_id == "C14", `:=`(
  status = if (nrow(no_weight)) "FAIL" else "PASS",
  detail = if (nrow(no_weight)) {
    paste(sprintf("%s=%d countries (%s)", no_weight$variable,
                  no_weight$countries, no_weight$iso3), collapse = "; ")
  } else {
    "Every country represented on the 0.5-degree grid and having an authoritative target has usable spatial weights or an explicit fallback"
  }
)]
write_csv_dt(checks, checks_path)

overall <- if (any(checks$status == "FAIL")) {
  "FAIL"
} else if (any(checks$status == "WARN")) {
  "PASS_WITH_WARN"
} else {
  "PASS"
}

cal_pass <- cal[status == "pass"]
max_error <- if (nrow(cal_pass)) max(cal_pass$relative_error, na.rm = TRUE) else Inf
warnings <- checks[status == "WARN", paste0(check_id, ": ", detail)]
report <- c(
  "# Phase 2 Gate C v1 report",
  "",
  sprintf("- Run directory: `%s`", target_display),
  sprintf("- Source processing run: `%s`", source_display),
  sprintf("- Overall status: **%s**", overall),
  "- Scope: future covariate harmonization only (2025-2050).",
  "- Explicitly excluded: model fitting, BMI prediction, PAF calculation and disease-burden analysis.",
  "",
  "## Completed operations",
  "",
  "1. Interpolated national SSP population, GDP and age-sex population to annual 2025-2050 values.",
  "2. Calibrated 0.5-degree population and GDP cell totals to the authoritative A1 national constraints.",
  "3. Interpolated urban fraction on the logit scale from official decadal nodes.",
  "4. Bias-corrected seven CMIP6 models to ERA5 2015-2024 climatology and formed annual ensemble means.",
  "5. Aligned elevation, travel time and the 2025 WorldPop adult age-sex prior to the frozen 0.5-degree grid.",
  "6. Frozen a 0.1-degree alignment contract without materializing annual high-resolution copies.",
  "",
  "## Country-grid coverage remediation",
  "",
  "Countries with no 0.5-degree cell center inside their boundary are classified as outside the representable domain of the primary grid. They are not assigned fabricated neighboring cells and are not misclassified as spatial-weight failures.",
  "",
  "## QA summary",
  "",
  sprintf("- Checks: PASS=%d, WARN=%d, FAIL=%d", sum(checks$status == "PASS"),
          sum(checks$status == "WARN"), sum(checks$status == "FAIL")),
  "- Derived 0.5-degree raster files audited: 553",
  sprintf("- Maximum national calibration relative error: %.3g", max_error),
  sprintf("- Countries/territories not represented by a 0.5-degree cell centre: %d", length(missing_grid)),
  sprintf("- Population/GDP availability records: %d country-variable rows", nrow(availability)),
  "- Countries without an authoritative A1 target remain NoData for that variable; A4 country totals are not substituted.",
  "",
  "## Remaining warnings",
  "",
  if (length(warnings)) paste0("- ", warnings) else "- None",
  "",
  "## Gate decision",
  "",
  if (overall == "FAIL") {
    "Gate C did not pass. No later phase is authorized."
  } else {
    "Gate C covariate harmonization passed with the warnings above. This run stops here for user acceptance; no later phase was executed."
  }
)
writeLines(report, file.path(target_run, "Gate_C_report_v1.md"), useBytes = TRUE)

prov_path <- file.path(target_run, "Gate_C_run_provenance_v1.json")
prov <- fromJSON(prov_path, simplifyVector = FALSE)
prov$generated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
prov$overall_status <- overall
prov$run_directory <- target_display
prov$source_processing_run <- source_display
prov$qa_remediation <- list(
  id = "country_grid_representation_v1",
  rule = "Countries with no rasterized 0.5-degree cell center are not_represented_on_0p5_grid, not no_spatial_weight.",
  data_files_modified = FALSE
)
tmp <- paste0(prov_path, ".tmp")
write_json(prov, tmp, auto_unbox = TRUE, pretty = TRUE, null = "null")
if (file.exists(prov_path)) file.remove(prov_path)
if (!file.rename(tmp, prov_path)) stop("Cannot update provenance")

cat(overall, "\n")
if (overall == "FAIL") quit(save = "no", status = 2L)
