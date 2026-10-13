if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

suppressPackageStartupMessages({
  library(data.table)
  library(countrycode)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: build_gate_b_country_crosswalk_v1.R <A1 population CSV> <A2 recode CSV> <output CSV>")
}

a1_path <- args[[1L]]
a2_recode_path <- args[[2L]]
out_path <- args[[3L]]

a1 <- fread(a1_path, select = "region")
a1_names <- sort(unique(a1$region))
a1_iso3 <- countrycode(a1_names, origin = "country.name", destination = "iso3c", warn = FALSE)

# Resolve a small set of current/official labels that differ from countrycode aliases.
manual_iso3 <- c(
  "Bahamas" = "BHS",
  "Bolivia" = "BOL",
  "Brunei" = "BRN",
  "Cape Verde" = "CPV",
  "Congo, Dem. Rep." = "COD",
  "Congo, Rep." = "COG",
  "Cote d'Ivoire" = "CIV",
  "Curaçao" = "CUW",
  "Egypt, Arab Rep." = "EGY",
  "Gambia, The" = "GMB",
  "Hong Kong SAR, China" = "HKG",
  "Iran, Islamic Rep." = "IRN",
  "Korea, Dem. People's Rep." = "PRK",
  "Korea, Rep." = "KOR",
  "Kyrgyz Republic" = "KGZ",
  "Lao PDR" = "LAO",
  "Macao SAR, China" = "MAC",
  "Micronesia" = "FSM",
  "Micronesia, Fed. Sts." = "FSM",
  "Moldova" = "MDA",
  "North Macedonia" = "MKD",
  "Kosovo" = "XKX",
  "Réunion" = "REU",
  "Russian Federation" = "RUS",
  "Slovak Republic" = "SVK",
  "Syrian Arab Republic" = "SYR",
  "Taiwan, China" = "TWN",
  "Tanzania" = "TZA",
  "Türkiye" = "TUR",
  "Venezuela, RB" = "VEN",
  "Vietnam" = "VNM",
  "West Bank and Gaza" = "PSE",
  "Yemen, Rep." = "YEM"
)

missing <- is.na(a1_iso3) & a1_names %in% names(manual_iso3)
a1_iso3[missing] <- unname(manual_iso3[a1_names[missing]])

a1_map <- data.table(
  dataset_id = "A1",
  source_id = a1_names,
  source_name = a1_names,
  iso3 = a1_iso3,
  id_system = "country name"
)

recode <- fread(a2_recode_path)
a2 <- recode[var == "region", .(source_id = varval, source_name = varvaldesc)]
a2[, m49 := suppressWarnings(as.integer(sub("^reg", "", source_id)))]
a2[, iso3 := countrycode(m49, origin = "un", destination = "iso3c", warn = FALSE)]
a2[source_name == "China, Taiwan Province of China", iso3 := "TWN"]
a2[, entity_type := fifelse(!is.na(iso3), "country_or_territory", "regional_aggregate")]
a2[, `:=`(dataset_id = "A2", id_system = "UN M49 embedded in regNNN")]
a2_map <- a2[, .(dataset_id, source_id, source_name, iso3, id_system, entity_type)]

a1_map[, entity_type := "country_or_territory"]

out <- rbindlist(list(a1_map, a2_map), use.names = TRUE, fill = TRUE)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
fwrite(out, out_path)

cat(sprintf("A1 rows=%d mapped=%d unmapped=%d\n", nrow(a1_map), sum(!is.na(a1_map$iso3)), sum(is.na(a1_map$iso3))))
cat(sprintf(
  "A2 rows=%d mapped_countries=%d regional_aggregates=%d\n",
  nrow(a2_map),
  sum(a2_map$entity_type == "country_or_territory"),
  sum(a2_map$entity_type == "regional_aggregate")
))
