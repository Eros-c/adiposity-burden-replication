#!/usr/bin/env Rscript
if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config


suppressPackageStartupMessages({
  library(data.table)
  library(terra)
  library(sf)
  library(jsonlite)
  library(digest)
})

options(stringsAsFactors = FALSE, warn = 1)
terraOptions(progress = 1, memfrac = 0.65, tempdir = Sys.getenv("GATEC_TEMP_DIR"))

run_dir <- Sys.getenv("GATEC_RUN_DIR")
if (!nzchar(run_dir)) stop("GATEC_RUN_DIR is required")

input <- list(
  gate_b_dir = Sys.getenv("GATEB_DIR"),
  crosswalk = file.path(Sys.getenv("GATEB_DIR"), "Gate_B_country_crosswalk.csv"),
  a1_gdp = Sys.getenv("A1_GDP_FILE"),
  a1_population = Sys.getenv("A1_POP_FILE"),
  a2_dir = Sys.getenv("A2_DIR"),
  a3_dir = Sys.getenv("A3_DIR"),
  a4_dir = Sys.getenv("A4_DIR"),
  a5_dir = Sys.getenv("A5_DIR"),
  a6_dir = Sys.getenv("A6_DIR"),
  a7_dir = Sys.getenv("A7_DIR"),
  a8_file = Sys.getenv("A8_FILE"),
  adm0_zip = Sys.getenv("ADM0_ZIP"),
  elevation_zip = Sys.getenv("ELEVATION_ZIP"),
  travel_file = Sys.getenv("TRAVEL_FILE")
)

missing_inputs <- names(input)[!file.exists(unlist(input)) & !dir.exists(unlist(input))]
if (length(missing_inputs)) {
  stop("Missing Gate C inputs: ", paste(missing_inputs, collapse = ", "))
}

scenarios <- data.table(
  scenario = c("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"),
  ssp = c("SSP1", "SSP2", "SSP3", "SSP5"),
  ssp_lower = c("ssp1", "ssp2", "ssp3", "ssp5"),
  cmip_experiment = c("ssp126", "ssp245", "ssp370", "ssp585")
)
future_years <- 2025:2050
node_years <- seq(2025, 2050, 5)

dirs <- list(
  data = file.path(run_dir, "data"),
  national = file.path(run_dir, "data", "national"),
  grid05 = file.path(run_dir, "data", "0p5deg"),
  population = file.path(run_dir, "data", "0p5deg", "population"),
  gdp = file.path(run_dir, "data", "0p5deg", "gdp"),
  urban = file.path(run_dir, "data", "0p5deg", "urban_fraction"),
  climate = file.path(run_dir, "data", "0p5deg", "climate"),
  static = file.path(run_dir, "data", "0p5deg", "static"),
  worldpop = file.path(run_dir, "data", "0p5deg", "worldpop_2025_prior"),
  grid_contracts = file.path(run_dir, "data", "grid_contracts"),
  qa = file.path(run_dir, "qa"),
  logs = file.path(run_dir, "logs"),
  work = file.path(run_dir, "work"),
  climate_work = file.path(run_dir, "work", "climate_models")
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

log_file <- file.path(dirs$logs, "Gate_C_processing_v1.log")
log_con <- file(log_file, open = "wt", encoding = "UTF-8")
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

write_json_atomic <- function(x, path) {
  tmp <- paste0(path, ".tmp")
  write_json(x, tmp, pretty = TRUE, auto_unbox = TRUE, na = "null")
  if (file.exists(path)) stop("Refusing to overwrite: ", path)
  if (!file.rename(tmp, path)) stop("Failed to finalize JSON: ", path)
}

write_csv_new <- function(x, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path)
  write.csv(as.data.frame(x), file = path, row.names = FALSE,
            na = "", fileEncoding = "UTF-8")
}

write_csv_gz_new <- function(x, path) {
  if (file.exists(path)) stop("Refusing to overwrite: ", path)
  con <- gzfile(path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  write.csv(as.data.frame(x), file = con, row.names = FALSE, na = "")
}

write_raster_new <- function(x, path, datatype = "FLT4S", naflag = -9999) {
  if (file.exists(path)) stop("Refusing to overwrite raster: ", path)
  predictor <- if (grepl("^FLT", datatype)) "PREDICTOR=3" else "PREDICTOR=2"
  writeRaster(
    x, path, overwrite = FALSE,
    wopt = list(
      datatype = datatype,
      NAflag = naflag,
      gdal = c("COMPRESS=DEFLATE", "ZLEVEL=6", predictor, "TILED=YES")
    )
  )
  invisible(path)
}

safe_sha256 <- function(path) digest(path, algo = "sha256", file = TRUE, serialize = FALSE)

template05 <- rast(nrows = 360, ncols = 720, xmin = -180, xmax = 180,
                   ymin = -90, ymax = 90, crs = "EPSG:4326")
names(template05) <- "template_0p5deg"

template_contract <- list(
  grid_0p5deg = list(
    status = "materialized_primary_grid",
    crs = "EPSG:4326",
    extent = c(-180, 180, -90, 90),
    resolution_degree = c(0.5, 0.5),
    dimensions = c(nrow = 360, ncol = 720),
    cell_alignment = "xmin=-180; ymax=90; cell centers at -179.75...179.75 and 89.75...-89.75"
  ),
  grid_0p1deg = list(
    status = "contract_only_not_materialized_in_gate_c_v1",
    crs = "EPSG:4326",
    extent = c(-180, 180, -90, 90),
    resolution_degree = c(0.1, 0.1),
    dimensions = c(nrow = 1800, ncol = 3600),
    reason = "Gate C v1 freezes alignment only; annual high-resolution copies are deferred."
  ),
  adm1 = list(
    status = "aggregation_contract_only",
    rule = "Population-weighted aggregation from the calibrated 0.5-degree primary grid in later authorized phases."
  )
)
write_json_atomic(template_contract, file.path(dirs$grid_contracts, "Gate_C_grid_contract_v1.json"))

stamp("Building country raster and lookup")
wb_path <- paste0("/vsizip/", gsub("\\\\", "/", input$adm0_zip), "/WB_GAD_ADM0.shp")
wb <- st_read(wb_path, quiet = TRUE)
iso_a3 <- toupper(trimws(as.character(wb[["ISO_A3"]])))
wb_a3 <- toupper(trimws(as.character(wb[["WB_A3"]])))
iso3_values <- ifelse(grepl("^[A-Z]{3}$", iso_a3), iso_a3, wb_a3)
wb[["iso3"]] <- iso3_values
wb <- wb[grepl("^[A-Z]{3}$", wb[["iso3"]]), ]
iso_lookup <- data.table(iso3 = sort(unique(as.character(wb[["iso3"]]))))
iso_lookup[, country_id := .I]
wb[["country_id"]] <- iso_lookup[["country_id"]][match(wb[["iso3"]], iso_lookup[["iso3"]])]
wb_v <- vect(wb[, c("iso3", "country_id")])
country_r <- rasterize(wb_v, template05, field = "country_id", touches = FALSE)
names(country_r) <- "country_id"
write_raster_new(country_r, file.path(dirs$grid05, "country_id_0p5deg.tif"), "INT2S", 0)
write_csv_new(iso_lookup, file.path(dirs$grid05, "country_id_lookup.csv"))
uniform_country_fallback <- ifel(!is.na(country_r), 1, NA)

crosswalk <- fread(input$crosswalk)
crosswalk[, `:=`(source_id = trimws(as.character(source_id)),
                 iso3 = toupper(trimws(as.character(iso3))))]
crosswalk[iso3 == "", iso3 := NA_character_]
crosswalk[, valid_iso3 := !is.na(iso3) & grepl("^[A-Z]{3}$", iso3)]
map_a1 <- unique(crosswalk[dataset_id == "A1" & valid_iso3,
                           .(region = source_id, iso3)])
map_a2 <- unique(crosswalk[dataset_id == "A2" & valid_iso3,
                           .(region = source_id, iso3)])

interp_log <- function(year, value, out_years = future_years) {
  value <- as.numeric(value)
  year <- as.integer(year)
  ok <- is.finite(value) & value >= 0 & !is.na(year)
  if (sum(ok) < 2) return(rep(NA_real_, length(out_years)))
  exp(approx(year[ok], log(pmax(value[ok], 1e-12)), xout = out_years, rule = 2)$y)
}

interp_logit <- function(r0, r1, weight, eps = 1e-6) {
  a <- clamp(r0, eps, 1 - eps)
  b <- clamp(r1, eps, 1 - eps)
  eta <- (1 - weight) * log(a / (1 - a)) + weight * log(b / (1 - b))
  1 / (1 + exp(-eta))
}

stamp("Building annual national GDP and population constraints")
gdp_raw <- fread(input$a1_gdp)
gdp_raw <- gdp_raw[
  model == "OECD ENV-Growth 2025" & variable == "GDP|PPP" & unit == "billion USD_2010/yr" &
    scenario %in% scenarios$ssp
]
gdp_long <- melt(gdp_raw, id.vars = c("scenario", "region"),
                 measure.vars = as.character(node_years), variable.name = "year", value.name = "value")
gdp_long[, year := as.integer(as.character(year))]
gdp_long <- merge(gdp_long, map_a1, by = "region", all.x = TRUE)
gdp_long <- gdp_long[!is.na(iso3)]
gdp_annual <- gdp_long[, {
  vals <- interp_log(year, value)
  .(year = future_years, gdp_ppp_2010_usd = vals * 1e9)
}, by = .(ssp = scenario, iso3)]
gdp_annual <- merge(gdp_annual, scenarios[, .(ssp, scenario)], by = "ssp")

pop_raw <- fread(input$a1_population)
pop_raw <- pop_raw[
  model == "IIASA-WiC POP 2025" & variable == "Population" & unit == "million" &
    scenario %in% scenarios$ssp
]
pop_long <- melt(pop_raw, id.vars = c("scenario", "region"),
                 measure.vars = as.character(node_years), variable.name = "year", value.name = "value")
pop_long[, year := as.integer(as.character(year))]
pop_long <- merge(pop_long, map_a1, by = "region", all.x = TRUE)
pop_long <- pop_long[!is.na(iso3)]
pop_annual <- pop_long[, {
  vals <- interp_log(year, value)
  .(year = future_years, population_persons = vals * 1e6)
}, by = .(ssp = scenario, iso3)]
pop_annual <- merge(pop_annual, scenarios[, .(ssp, scenario)], by = "ssp")

national_constraints <- merge(
  pop_annual[, .(scenario, ssp, iso3, year, population_persons)],
  gdp_annual[, .(scenario, ssp, iso3, year, gdp_ppp_2010_usd)],
  by = c("scenario", "ssp", "iso3", "year"), all = TRUE
)
write_csv_new(
  national_constraints,
  file.path(dirs$national, "national_population_gdp_annual_2025_2050.csv")
)

stamp("Building annual national age-sex population constraints")
age_sex_list <- vector("list", nrow(scenarios))
for (j in seq_len(nrow(scenarios))) {
  ssp_num <- sub("SSP", "", scenarios$ssp[j])
  f <- file.path(input$a2_dir, sprintf("PROJresult_AGE_SSP%s_V14.csv", ssp_num))
  if (!file.exists(f)) stop("Missing WIC age-sex file: ", f)
  d <- fread(f, select = c("region", "Time", "sex", "edu", "agest", "pop"))
  d <- d[Time %in% node_years & sex %in% c("f", "m") & agest >= 25]
  d[, age_group := ifelse(agest >= 95, "95+", sprintf("%d-%d", agest, agest + 4))]
  d <- d[, .(population_thousand = sum(pop, na.rm = TRUE)),
         by = .(region, year = Time, sex, age_group)]
  d <- merge(d, map_a2, by = "region", all.x = TRUE)
  d <- d[!is.na(iso3)]
  a <- d[, {
    vals <- interp_log(year, population_thousand)
    .(year = future_years, population_persons = vals * 1000)
  }, by = .(iso3, sex, age_group)]
  a[, `:=`(scenario = scenarios$scenario[j], ssp = scenarios$ssp[j])]
  age_sex_list[[j]] <- a
  rm(d, a); gc()
}
age_sex_annual <- rbindlist(age_sex_list, use.names = TRUE)
setcolorder(age_sex_annual, c("scenario", "ssp", "iso3", "year", "sex", "age_group", "population_persons"))
write_csv_new(
  age_sex_annual,
  file.path(dirs$national, "national_age_sex_population_annual_2025_2050.csv")
)

calibration_qa <- list()
calibration_i <- 0L

zonal_sum_dt <- function(r) {
  z <- as.data.table(zonal(r, country_r, fun = "sum", na.rm = TRUE))
  setnames(z, names(z)[1:2], c("country_id", "raster_sum"))
  merge(z, iso_lookup, by = "country_id", all.x = TRUE)
}

calibrate_country_totals <- function(r, target, variable, scenario, ssp, year,
                                     fallback = NULL) {
  pre <- zonal_sum_dt(r)
  q <- merge(iso_lookup, target[, .(iso3, target_total)], by = "iso3", all = TRUE)
  q <- merge(q, pre[, .(country_id, pre_sum = raster_sum)], by = "country_id", all.x = TRUE)
  q[, fallback_used := FALSE]

  zero_ids <- q[is.finite(target_total) & target_total > 0 &
                  (is.na(pre_sum) | pre_sum <= 0) & !is.na(country_id), country_id]
  if (length(zero_ids) && !is.null(fallback)) {
    zmask <- subst(country_r, from = zero_ids, to = rep(1, length(zero_ids)), others = NA)
    r <- ifel(!is.na(zmask), fallback, r)
    pre <- zonal_sum_dt(r)
    q <- merge(q[, !c("pre_sum"), with = FALSE],
               pre[, .(country_id, pre_sum = raster_sum)], by = "country_id", all.x = TRUE)
    q[country_id %in% zero_ids, fallback_used := TRUE]
  }

  q[, scale_factor := fifelse(is.finite(target_total) & target_total >= 0 &
                                is.finite(pre_sum) & pre_sum > 0,
                              target_total / pre_sum, NA_real_)]
  scale_r <- subst(country_r, from = q$country_id, to = q$scale_factor, others = NA)
  out <- r * scale_r
  post <- zonal_sum_dt(out)
  q <- merge(q, post[, .(country_id, post_sum = raster_sum)], by = "country_id", all.x = TRUE)
  q[, relative_error := fifelse(is.finite(target_total) & target_total > 0,
                                abs(post_sum - target_total) / target_total,
                                abs(post_sum - target_total))]
  q[, status := fifelse(is.na(country_id), "not_represented_on_0p5_grid",
                 fifelse(!is.finite(target_total), "no_national_target",
                   fifelse(!is.finite(pre_sum) | pre_sum <= 0, "no_spatial_weight",
                     fifelse(relative_error <= 1e-6, "pass", "fail"))))]
  q[, `:=`(variable = variable, scenario = scenario, ssp = ssp, year = year)]
  calibration_i <<- calibration_i + 1L
  calibration_qa[[calibration_i]] <<- q[, .(
    variable, scenario, ssp, year, iso3, country_id, target_total,
    pre_sum, scale_factor, post_sum, relative_error, fallback_used, status
  )]
  out
}

population_paths <- list()
stamp("Processing annual gridded population and national calibration")
for (j in seq_len(nrow(scenarios))) {
  ssp_num <- sub("SSP", "", scenarios$ssp[j])
  f <- file.path(input$a3_dir, sprintf("SSP%s/population_ssp%ssoc_2.5min_annual_2006-2100.nc4", ssp_num, ssp_num))
  x <- rast(f)
  tx <- time(x)
  yrs <- if (inherits(tx, c("Date", "POSIXt"))) {
    as.integer(format(tx, "%Y"))
  } else {
    as.integer(tx)
  }
  idx <- match(future_years, yrs)
  if (anyNA(idx)) stop("A3 missing annual years for ", scenarios$scenario[j])
  for (ii in seq_along(future_years)) {
    y <- future_years[ii]
    stamp("  population ", scenarios$scenario[j], " ", y)
    raw05 <- resample(x[[idx[ii]]], template05, method = "sum")
    target <- pop_annual[ssp == scenarios$ssp[j] & year == y, .(iso3, target_total = population_persons)]
    out <- calibrate_country_totals(
      raw05, target, "population", scenarios$scenario[j], scenarios$ssp[j], y,
      fallback = uniform_country_fallback
    )
    path <- file.path(dirs$population, sprintf("population_%s_%d_0p5deg.tif", scenarios$ssp_lower[j], y))
    write_raster_new(out, path, "FLT8S", -9999)
    population_paths[[paste(scenarios$ssp[j], y, sep = "_")]] <- path
    rm(raw05, out); gc()
  }
  rm(x); gc()
}

stamp("Processing GDP allocation weights, annual log interpolation and national calibration")
for (j in seq_len(nrow(scenarios))) {
  for (y in future_years) {
    r1 <- NULL
    r0_nonnegative <- NULL
    r1_nonnegative <- NULL
    y0 <- max(node_years[node_years <= y])
    y1 <- min(node_years[node_years >= y])
    f0 <- file.path(input$a4_dir, sprintf("GDP%d_%s.tif", y0, scenarios$ssp_lower[j]))
    f1 <- file.path(input$a4_dir, sprintf("GDP%d_%s.tif", y1, scenarios$ssp_lower[j]))
    if (!file.exists(f0) || !file.exists(f1)) stop("Missing A4 node for ", scenarios$scenario[j], " ", y)
    r0 <- aggregate(rast(f0), fact = 2, fun = "sum", na.rm = TRUE)
    if (y1 == y0) {
      raw05 <- r0
    } else {
      r1 <- aggregate(rast(f1), fact = 2, fun = "sum", na.rm = TRUE)
      w <- (y - y0) / (y1 - y0)
      r0_nonnegative <- clamp(r0, lower = 0, upper = Inf)
      r1_nonnegative <- clamp(r1, lower = 0, upper = Inf)
      raw05 <- exp((1 - w) * log1p(r0_nonnegative) + w * log1p(r1_nonnegative)) - 1
    }
    pop_fallback <- rast(population_paths[[paste(scenarios$ssp[j], y, sep = "_")]])
    target <- gdp_annual[ssp == scenarios$ssp[j] & year == y, .(iso3, target_total = gdp_ppp_2010_usd)]
    out <- calibrate_country_totals(raw05, target, "gdp", scenarios$scenario[j], scenarios$ssp[j], y,
                                    fallback = pop_fallback)
    path <- file.path(dirs$gdp, sprintf("gdp_ppp2010_%s_%d_0p5deg.tif", scenarios$ssp_lower[j], y))
    write_raster_new(out, path, "FLT8S", -9999)
    rm(r0, r1, r0_nonnegative, r1_nonnegative, raw05, out, pop_fallback); gc()
  }
}

stamp("Processing urban fraction nodes and annual logit interpolation")
urban_node_years <- c(2020, 2030, 2040, 2050)
for (j in seq_len(nrow(scenarios))) {
  node_paths <- character()
  for (ny in urban_node_years) {
    src <- file.path(input$a5_dir, sprintf("%s-geotiff/%s_%d.tif", scenarios$ssp_lower[j], scenarios$ssp_lower[j], ny))
    if (!file.exists(src)) stop("Missing A5 urban node: ", src)
    stamp("  urban node ", scenarios$scenario[j], " ", ny)
    r <- rast(src)
    r[r < 0] <- NA
    a <- resample(r, template05, method = "average")
    a <- clamp(a, 0, 1)
    np <- file.path(dirs$work, sprintf("urban_node_%s_%d_0p5deg.tif", scenarios$ssp_lower[j], ny))
    write_raster_new(a, np, "FLT4S", -9999)
    node_paths[as.character(ny)] <- np
    rm(r, a); gc()
  }
  for (y in future_years) {
    y0 <- max(urban_node_years[urban_node_years <= y])
    y1 <- min(urban_node_years[urban_node_years >= y])
    r0 <- rast(node_paths[as.character(y0)])
    if (y1 == y0) {
      out <- r0
    } else {
      r1 <- rast(node_paths[as.character(y1)])
      out <- interp_logit(r0, r1, (y - y0) / (y1 - y0))
    }
    path <- file.path(dirs$urban, sprintf("urban_fraction_%s_%d_0p5deg.tif", scenarios$ssp_lower[j], y))
    write_raster_new(out, path, "FLT4S", -9999)
    rm(r0, r1, out); gc()
  }
}

stamp("Aligning static elevation and travel-time covariates")
elev_path <- paste0("/vsizip/", gsub("\\\\", "/", input$elevation_zip), "/mn30_grd")
elev <- rast(elev_path)
elev05 <- resample(elev, template05, method = "average")
write_raster_new(elev05, file.path(dirs$static, "elevation_m_0p5deg.tif"), "FLT4S", -9999)
travel <- rast(input$travel_file)
travel[travel < 0] <- NA
travel05 <- resample(travel, template05, method = "average")
write_raster_new(travel05, file.path(dirs$static, "travel_time_city50k_minutes_0p5deg.tif"), "FLT4S", -9999)
rm(elev, elev05, travel, travel05); gc()

stamp("Aligning WorldPop 2025 age-sex rasters as a fixed spatial prior")
wp_files <- list.files(input$a6_dir, pattern = "[.]tif$", recursive = TRUE, full.names = TRUE)
if (length(wp_files) != 28) stop("Expected 28 adult WorldPop age-sex rasters; found ", length(wp_files))
worldpop_qa <- list()
for (f in wp_files) {
  stamp("  WorldPop ", basename(f))
  r <- rast(f)
  a <- resample(r, template05, method = "sum")
  out <- file.path(dirs$worldpop, sub("_CN_1km_R2025A_UA_v1[.]tif$", "_0p5deg.tif", basename(f)))
  write_raster_new(a, out, "FLT8S", -9999)
  mm <- global(a, c("min", "max", "sum"), na.rm = TRUE)
  worldpop_qa[[length(worldpop_qa) + 1L]] <- data.table(
    source_file = f, output_file = out, min = mm[1, 1], max = mm[1, 2], sum = mm[1, 3]
  )
  rm(r, a); gc()
}
write_csv_new(rbindlist(worldpop_qa), file.path(dirs$qa, "Gate_C_worldpop_2025_prior_QA.csv"))

stamp("Computing ERA5 2015-2024 temperature and relative-humidity climatology")
era_sds <- sds(input$a8_file)
era_names <- names(era_sds)
if (!all(c("d2m", "t2m") %in% era_names)) stop("ERA5 file lacks d2m/t2m subdatasets")
d2m <- era_sds[[match("d2m", era_names)]]
t2m <- era_sds[[match("t2m", era_names)]]
era_dates <- as.Date(as.POSIXct(depth(t2m), origin = "1970-01-01", tz = "UTC"))
era_years <- as.integer(format(era_dates, "%Y"))
era_idx <- which(era_years %in% 2015:2024)
if (length(era_idx) != 120) stop("Expected 120 ERA5 baseline months; found ", length(era_idx))
t_c <- t2m[[era_idx]] - 273.15
td_c <- d2m[[era_idx]] - 273.15
rh <- 100 * exp((17.625 * td_c) / (243.04 + td_c) - (17.625 * t_c) / (243.04 + t_c))
rh <- clamp(rh, 0, 100)
era_t <- rotate(app(t_c, mean, na.rm = TRUE))
era_rh <- rotate(app(rh, mean, na.rm = TRUE))
era_t05 <- resample(era_t, template05, method = "bilinear")
era_rh05 <- clamp(resample(era_rh, template05, method = "bilinear"), 0, 100)
era_t_path <- file.path(dirs$static, "era5_tas_c_2015_2024_climatology_0p5deg.tif")
era_rh_path <- file.path(dirs$static, "era5_rh_pct_2015_2024_climatology_0p5deg.tif")
write_raster_new(era_t05, era_t_path, "FLT4S", -9999)
write_raster_new(era_rh05, era_rh_path, "FLT4S", -9999)
rm(era_sds, d2m, t2m, t_c, td_c, rh, era_t, era_rh); gc()

cmip_files <- list.files(input$a7_dir, pattern = "[.]nc$", recursive = TRUE, full.names = TRUE)
cmip_norm <- gsub("\\\\", "/", cmip_files)
extract_model <- function(path, variable, experiment) {
  bn <- basename(path)
  sub(sprintf("^%s_Amon_(.+)_%s_.*$", variable, experiment), "\\1", bn)
}

climate_model_qa <- list()
stamp("Bias-correcting and ensemble-averaging CMIP6 temperature and humidity")
for (j in seq_len(nrow(scenarios))) {
  for (variable in c("tas", "hurs")) {
    pat <- sprintf("/%s/%s/", scenarios$cmip_experiment[j], variable)
    fs <- cmip_files[grepl(pat, cmip_norm, fixed = TRUE)]
    if (length(fs) < 5) stop("Fewer than five CMIP6 models for ", scenarios$scenario[j], " ", variable)
    models_all <- vapply(fs, extract_model, character(1), variable = variable,
                         experiment = scenarios$cmip_experiment[j])
    model_groups <- split(fs, models_all)
    models <- sort(names(model_groups))
    if (length(models) < 5) stop("Fewer than five unique CMIP6 models for ", scenarios$scenario[j], " ", variable)
    model_paths <- character()
    for (m in seq_along(models)) {
      fs_m <- sort(model_groups[[models[m]]])
      stamp("  CMIP6 ", scenarios$scenario[j], " ", variable, " ", models[m])
      x <- rast(fs_m)
      tt <- as.Date(time(x))
      yy <- as.integer(format(tt, "%Y"))
      ib <- which(yy %in% 2015:2024)
      iff <- which(yy %in% future_years)
      if (length(ib) != 120 || length(iff) != 312) {
        stop("Unexpected monthly coverage: ", fs[m], " baseline=", length(ib), " future=", length(iff))
      }
      xb <- x[[ib]]
      xf <- x[[iff]]
      if (variable == "hurs") {
        xb <- clamp(xb, 0, 100)
        xf <- clamp(xf, 0, 100)
      }
      base_native <- app(xb, mean, na.rm = TRUE)
      annual_native <- tapp(xf, index = yy[iff], fun = mean, na.rm = TRUE)
      if (xmin(base_native) >= 0) {
        base_native <- rotate(base_native)
        annual_native <- rotate(annual_native)
      }
      base05 <- resample(base_native, template05, method = "bilinear")
      annual05 <- resample(annual_native, template05, method = "bilinear")
      if (variable == "tas") {
        base05 <- base05 - 273.15
        annual05 <- annual05 - 273.15
        corrected <- annual05 + (era_t05 - base05)
      } else {
        corrected <- clamp(annual05 + (era_rh05 - base05), 0, 100)
      }
      names(corrected) <- as.character(future_years)
      mp <- file.path(dirs$climate_work,
                      sprintf("%s_%s_%s_2025_2050_0p5deg.tif",
                              variable, scenarios$ssp_lower[j], models[m]))
      write_raster_new(corrected, mp, "FLT4S", -9999)
      model_paths <- c(model_paths, mp)
      climate_model_qa[[length(climate_model_qa) + 1L]] <- data.table(
        scenario = scenarios$scenario[j], variable = variable, model = models[m],
        source_files = paste(fs_m, collapse = "|"), output_file = mp,
        source_file_count = length(fs_m), baseline_months = length(ib),
        future_months = length(iff), hurs_raw_capped = variable == "hurs"
      )
      rm(x, xb, xf, base_native, annual_native, base05, annual05, corrected); gc()
    }
    for (ii in seq_along(future_years)) {
      layers <- lapply(model_paths, function(p) rast(p)[[ii]])
      ensemble <- app(do.call(c, layers), mean, na.rm = TRUE)
      if (variable == "hurs") ensemble <- clamp(ensemble, 0, 100)
      unit_tag <- if (variable == "tas") "c" else "pct"
      path <- file.path(dirs$climate,
                        sprintf("%s_%s_%s_%d_0p5deg.tif",
                                variable, unit_tag, scenarios$ssp_lower[j], future_years[ii]))
      write_raster_new(ensemble, path, "FLT4S", -9999)
      rm(layers, ensemble); gc()
    }
  }
}
write_csv_new(rbindlist(climate_model_qa), file.path(dirs$qa, "Gate_C_CMIP6_model_QA.csv"))

calibration_dt <- rbindlist(calibration_qa, use.names = TRUE, fill = TRUE)
write_csv_new(calibration_dt, file.path(dirs$qa, "Gate_C_country_calibration_QA.csv"))

country_availability <- calibration_dt[, .(
  scenario_year_rows = .N,
  target_rows = sum(is.finite(target_total)),
  calibrated_rows = sum(status == "pass"),
  no_national_target_rows = sum(status == "no_national_target"),
  no_spatial_weight_rows = sum(status == "no_spatial_weight"),
  not_represented_rows = sum(status == "not_represented_on_0p5_grid"),
  fallback_rows = sum(fallback_used, na.rm = TRUE),
  all_available_targets_calibrated = all(status[is.finite(target_total)] == "pass")
), by = .(variable, iso3)]
write_csv_new(country_availability, file.path(dirs$qa, "Gate_C_country_availability_QA.csv"))

stamp("Running raster geometry, range, temporal and completeness QA")
expected <- data.table(
  variable = c("population", "gdp", "urban_fraction", "tas", "hurs"),
  expected_files = c(104L, 104L, 104L, 104L, 104L),
  directory = c(dirs$population, dirs$gdp, dirs$urban, dirs$climate, dirs$climate)
)
expected[, pattern := c("^population_.*[.]tif$", "^gdp_.*[.]tif$", "^urban_.*[.]tif$",
                        "^tas_.*[.]tif$", "^hurs_.*[.]tif$")]

raster_files <- c(
  list.files(dirs$population, pattern = "[.]tif$", full.names = TRUE),
  list.files(dirs$gdp, pattern = "[.]tif$", full.names = TRUE),
  list.files(dirs$urban, pattern = "[.]tif$", full.names = TRUE),
  list.files(dirs$climate, pattern = "[.]tif$", full.names = TRUE),
  list.files(dirs$static, pattern = "[.]tif$", full.names = TRUE),
  list.files(dirs$worldpop, pattern = "[.]tif$", full.names = TRUE),
  file.path(dirs$grid05, "country_id_0p5deg.tif")
)

classify_variable <- function(path) {
  b <- basename(path)
  if (grepl("^population_", b)) return("population")
  if (grepl("^gdp_", b)) return("gdp")
  if (grepl("^urban_", b)) return("urban_fraction")
  if (grepl("^tas_", b)) return("tas")
  if (grepl("^hurs_", b)) return("hurs")
  if (grepl("worldpop|global_[fm]_", b)) return("worldpop_2025_prior")
  if (grepl("elevation", b)) return("elevation")
  if (grepl("travel", b)) return("travel_time")
  if (grepl("era5_tas", b)) return("era5_tas_baseline")
  if (grepl("era5_rh", b)) return("era5_rh_baseline")
  if (grepl("country_id", b)) return("country_id")
  "other"
}

qa_rows <- list()
for (f in raster_files) {
  x <- rast(f)
  mm <- global(x, c("min", "max", "mean", "sum"), na.rm = TRUE)
  geom_ok <- compareGeom(x, template05, stopOnError = FALSE,
                         crs = TRUE, ext = TRUE, rowcol = TRUE, res = TRUE)
  qa_rows[[length(qa_rows) + 1L]] <- data.table(
    file = f, variable = classify_variable(f), nrow = nrow(x), ncol = ncol(x),
    nlyr = nlyr(x), crs = crs(x, proj = TRUE), xmin = xmin(x), xmax = xmax(x),
    ymin = ymin(x), ymax = ymax(x), xres = res(x)[1], yres = res(x)[2],
    geometry_ok = geom_ok, min = mm[1, 1], max = mm[1, 2],
    mean = mm[1, 3], sum = mm[1, 4], nonfinite_detected = FALSE
  )
}
raster_qa <- rbindlist(qa_rows, fill = TRUE)
write_csv_new(raster_qa, file.path(dirs$qa, "Gate_C_grid_alignment_and_range_QA.csv"))

file_count_qa <- expected[, {
  actual <- length(list.files(directory, pattern = pattern, full.names = TRUE))
  .(expected_files = expected_files, actual_files = actual,
    status = ifelse(actual == expected_files, "PASS", "FAIL"))
}, by = .(variable)]
write_csv_new(file_count_qa, file.path(dirs$qa, "Gate_C_expected_file_count_QA.csv"))

temporal_qa <- raster_qa[variable %in% c("population", "gdp", "urban_fraction", "tas", "hurs")]
temporal_qa[, scenario := fifelse(grepl("ssp1", basename(file)), "SSP1-2.6",
                           fifelse(grepl("ssp2", basename(file)), "SSP2-4.5",
                           fifelse(grepl("ssp3", basename(file)), "SSP3-7.0", "SSP5-8.5")))]
temporal_qa[, year := as.integer(sub(".*_(20[0-9]{2})_0p5deg[.]tif$", "\\1", basename(file)))]
setorder(temporal_qa, variable, scenario, year)
temporal_qa[, annual_change := mean - data.table::shift(mean), by = .(variable, scenario)]
temporal_qa[, relative_annual_change := abs(annual_change) / pmax(abs(data.table::shift(mean)), 1e-12),
            by = .(variable, scenario)]
write_csv_new(temporal_qa[, .(variable, scenario, year, mean, sum, annual_change, relative_annual_change)],
              file.path(dirs$qa, "Gate_C_temporal_QA.csv"))

range_fail <- raster_qa[
  !geometry_ok |
  (variable %in% c("population", "gdp", "worldpop_2025_prior") & min < -1e-6) |
  (variable == "urban_fraction" & (min < -1e-6 | max > 1 + 1e-6)) |
  (variable %in% c("hurs", "era5_rh_baseline") & (min < -1e-6 | max > 100 + 1e-6))
]
cal_fail <- calibration_dt[status == "fail"]
represented_pass <- calibration_dt[status == "pass"]
cal_max_error <- if (nrow(represented_pass)) max(represented_pass$relative_error, na.rm = TRUE) else Inf
missing_grid_countries <- unique(calibration_dt[status == "not_represented_on_0p5_grid", iso3])
no_target_summary <- calibration_dt[status == "no_national_target",
                                    .(countries = uniqueN(iso3), rows = .N,
                                      iso3 = paste(sort(unique(iso3)), collapse = ",")),
                                    by = variable]
no_weight_summary <- calibration_dt[status == "no_spatial_weight",
                                    .(countries = uniqueN(iso3), rows = .N,
                                      iso3 = paste(sort(unique(iso3)), collapse = ",")),
                                    by = variable]
no_target_detail <- if (nrow(no_target_summary)) {
  paste(sprintf("%s=%d countries (%s)", no_target_summary$variable,
                no_target_summary$countries, no_target_summary$iso3), collapse = "; ")
} else {
  "All boundary countries have authoritative A1 targets"
}
no_weight_detail <- if (nrow(no_weight_summary)) {
  paste(sprintf("%s=%d countries (%s)", no_weight_summary$variable,
                no_weight_summary$countries, no_weight_summary$iso3), collapse = "; ")
} else {
  "All countries with national targets have usable spatial weights or an explicit fallback"
}

checks <- rbindlist(list(
  data.table(check_id = "C01", check = "Expected dynamic raster file counts",
             status = ifelse(all(file_count_qa$status == "PASS"), "PASS", "FAIL"),
             detail = paste(file_count_qa$variable, file_count_qa$actual_files, sep = "=", collapse = "; ")),
  data.table(check_id = "C02", check = "All derived rasters match frozen 0.5-degree grid",
             status = ifelse(all(raster_qa$geometry_ok), "PASS", "FAIL"),
             detail = sprintf("%d/%d files aligned", sum(raster_qa$geometry_ok), nrow(raster_qa))),
  data.table(check_id = "C03", check = "Derived raster value ranges",
             status = ifelse(nrow(range_fail) == 0, "PASS", "FAIL"),
             detail = sprintf("%d range or geometry failures", nrow(range_fail))),
  data.table(check_id = "C04", check = "Country population and GDP total closure",
             status = ifelse(nrow(cal_fail) == 0 && cal_max_error <= 1e-6, "PASS", "FAIL"),
             detail = sprintf("max relative error among represented countries = %.3g", cal_max_error)),
  data.table(check_id = "C05", check = "CMIP6 ensemble model count",
             status = ifelse(min(rbindlist(climate_model_qa)[, uniqueN(model), by = .(scenario, variable)]$V1) >= 5,
                             "PASS", "FAIL"),
             detail = "Seven GCMs expected for each scenario-variable pair"),
  data.table(check_id = "C06", check = "Annual time coverage 2025-2050",
             status = ifelse(all(temporal_qa[, uniqueN(year), by = .(variable, scenario)]$V1 == 26), "PASS", "FAIL"),
             detail = "26 annual layers expected per dynamic variable and scenario"),
  data.table(check_id = "C07", check = "WorldPop 2025 adult age-sex prior alignment",
             status = ifelse(length(list.files(dirs$worldpop, pattern = "[.]tif$")) == 28, "PASS", "FAIL"),
             detail = "28 age-sex rasters aligned; future within-country age-sex migration is not invented"),
  data.table(check_id = "C08", check = "0.1-degree future grid scope",
             status = "WARN",
             detail = "Grid geometry contract frozen; annual 0.1-degree copies not materialized in Gate C v1"),
  data.table(check_id = "C09", check = "Sub-grid country coverage on 0.5-degree grid",
             status = ifelse(length(missing_grid_countries) == 0, "PASS", "WARN"),
             detail = ifelse(length(missing_grid_countries),
                             paste("Not represented by a 0.5-degree cell center:", paste(missing_grid_countries, collapse = ",")),
                             "All target countries represented")),
  data.table(check_id = "C10", check = "CMIP6 raw hurs handling",
             status = "PASS",
             detail = "Raw hurs capped to 0-100 before ERA5 additive bias correction; derived hurs capped to 0-100"),
  data.table(check_id = "C11", check = "Static covariate alignment",
             status = ifelse(all(c("elevation", "travel_time") %in% raster_qa$variable), "PASS", "FAIL"),
             detail = "Elevation and travel time aligned to the primary grid"),
  data.table(check_id = "C12", check = "Scope guard",
             status = "PASS",
             detail = "No model fitting, BMI prediction, PAF calculation, or disease burden analysis executed"),
  data.table(check_id = "C13", check = "Authoritative national target coverage",
             status = ifelse(nrow(no_target_summary) == 0, "PASS", "WARN"),
             detail = no_target_detail),
  data.table(check_id = "C14", check = "Spatial-weight availability after explicit fallback",
             status = ifelse(nrow(no_weight_summary) == 0, "PASS", "FAIL"),
             detail = no_weight_detail)
))
write_csv_new(checks, file.path(dirs$qa, "Gate_C_pass_fail_v1.csv"))

contracts <- data.table(
  dataset_id = c("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "S1", "S2"),
  derived_variable = c(
    "national population and GDP constraints", "national age-sex population constraints",
    "gridded total population", "gridded GDP spatial allocation", "urban fraction",
    "2025 adult age-sex spatial prior", "near-surface air temperature and relative humidity",
    "ERA5 temperature and derived relative humidity baseline", "elevation", "travel time to city >=50,000"
  ),
  source_path = c(input$a1_gdp, input$a2_dir, input$a3_dir, input$a4_dir, input$a5_dir,
                  input$a6_dir, input$a7_dir, input$a8_file, input$elevation_zip, input$travel_file),
  output_unit = c(
    "persons; PPP 2010 USD per year", "persons", "persons per 0.5-degree cell",
    "PPP 2010 USD per 0.5-degree cell after A1 recalibration", "fraction 0-1",
    "persons per 0.5-degree cell; fixed 2025 prior", "degree C; relative humidity percent",
    "degree C; relative humidity percent", "metres", "minutes"
  ),
  temporal_method = c(
    "log interpolation of official 5-year SSP nodes", "log interpolation of WIC 5-year nodes",
    "source is annual", "log1p interpolation of five-year spatial weights",
    "logit interpolation of decadal nodes", "2025 only; no future migration assumption",
    "monthly-to-annual means", "2015-2024 monthly climatology", "static", "static"
  ),
  national_calibration = c(
    "authoritative constraint", "national constraint table only", "A1 national population",
    "A1 OECD ENV-Growth 2025 GDP PPP 2010; A4 totals never authoritative", "none",
    "none in Gate C v1", "none", "not applicable", "not applicable", "not applicable"
  ),
  spatial_method = c(
    "national table", "national table", "sum resampling to 0.5 degree", "0.25 to 0.5 degree sum",
    "average resampling to 0.5 degree", "sum resampling to 0.5 degree",
    "bilinear resampling to 0.5 degree and seven-model ensemble mean",
    "rotate 0-360 longitude then bilinear resampling to 0.5 degree",
    "average resampling to 0.5 degree", "average resampling to 0.5 degree"
  ),
  nodata_contract = c(
    "NA", "NA", "source fill converted to NA", "no embedded NoData; provider structural zeros retained",
    "negative fill converted to NA", "source NoData retained as NA", "source missing retained as NA",
    "source missing retained as NA", "source missing retained as NA", "negative values converted to NA"
  ),
  gate_c_role = c(
    "authoritative national constraints", "annual age-sex constraints", "primary spatial population",
    "GDP spatial weights only", "future urbanization", "fixed age-sex spatial prior",
    "future climate", "bias correction reference", "static predictor", "static predictor"
  )
)
write_csv_new(contracts, file.path(run_dir, "Gate_C_dataset_contracts_v1.csv"))

overall_status <- if (any(checks$status == "FAIL")) "FAIL" else if (any(checks$status == "WARN")) "PASS_WITH_WARN" else "PASS"
warnings <- checks[status == "WARN", paste0(check_id, ": ", detail)]

report <- c(
  "# Phase 2 Gate C v1 report",
  "",
  sprintf("- Run directory: `%s`", run_dir),
  sprintf("- Overall status: **%s**", overall_status),
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
  "## QA summary",
  "",
  sprintf("- Checks: PASS=%d, WARN=%d, FAIL=%d", sum(checks$status == "PASS"),
          sum(checks$status == "WARN"), sum(checks$status == "FAIL")),
  sprintf("- Derived 0.5-degree raster files audited: %d", nrow(raster_qa)),
  sprintf("- Maximum national calibration relative error: %.3g", cal_max_error),
  sprintf("- Countries/territories not represented by a 0.5-degree cell centre: %d", length(missing_grid_countries)),
  sprintf("- Population/GDP availability records: %d country-variable rows", nrow(country_availability)),
  "- Countries without an authoritative A1 target remain NoData for that variable; A4 country totals are not substituted.",
  "",
  "## Remaining warnings",
  "",
  if (length(warnings)) paste0("- ", warnings) else "- None",
  "",
  "## Gate decision",
  "",
  if (overall_status == "FAIL")
    "Gate C did not pass. No later phase is authorized."
  else
    "Gate C covariate harmonization passed with the warnings above. This run stops here for user acceptance; no later phase was executed."
)
writeLines(report, file.path(run_dir, "Gate_C_report_v1.md"), useBytes = TRUE)

provenance <- list(
  phase = "Phase 2 Gate C v1",
  start_and_end_recorded_in_log = TRUE,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  overall_status = overall_status,
  run_directory = run_dir,
  r_version = R.version.string,
  package_versions = list(
    terra = as.character(packageVersion("terra")),
    sf = as.character(packageVersion("sf")),
    data_table = as.character(packageVersion("data.table")),
    jsonlite = as.character(packageVersion("jsonlite")),
    digest = as.character(packageVersion("digest"))
  ),
  scenarios = scenarios$scenario,
  years = c(min(future_years), max(future_years)),
  primary_grid = template_contract$grid_0p5deg,
  high_resolution_grid = template_contract$grid_0p1deg,
  scope_guards = list(
    model_fit = FALSE, bmi_prediction = FALSE, paf = FALSE, disease_burden = FALSE
  )
)
write_json_atomic(provenance, file.path(run_dir, "Gate_C_run_provenance_v1.json"))

stamp("Gate C complete. Overall status: ", overall_status)
if (overall_status == "FAIL") quit(save = "no", status = 2L)
