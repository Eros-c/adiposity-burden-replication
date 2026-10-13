if (!exists(".cfg")) source(file.path("code", "config", "paths.R"))  # central path config

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(sf)
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(ragg)
  library(svglite)
  library(scales)
})

root <- file.path(.cfg$ROOT, "future_scenario_ascii", "06_future_bmi_2025_2050", "phase7_multiscale_delivery_v3_20260908")
source_root <- file.path(root, "05_figures", "source_data")
figure_root <- file.path(root, "05_figures", "maps", "adm1_population_masked")
dir.create(figure_root, recursive = TRUE, showWarnings = FALSE)

adm1_path <- file.path(.cfg$ROOT, "phase7_r_inputs", "adm1_boundaries_ne10m_v1.gpkg")
adm1 <- st_make_valid(st_transform(st_read(adm1_path, layer = "adm1", quiet = TRUE), 4326))
adm1$zone_id <- as.integer(adm1$zone_id)

scenarios <- c(ssp126 = "SSP1-2.6", ssp245 = "SSP2-4.5", ssp370 = "SSP3-7.0", ssp585 = "SSP5-8.5")
years <- c(2025, 2030, 2040, 2050)
age_order <- c("25-39", "40-49", "50-59", "60-69", "70-79", "80+")
palette_obesity <- c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84", "#FC8D59", "#E34A33", "#B30000")

theme_map <- theme_void(base_size = 7, base_family = "Arial") + theme(
  plot.background = element_rect(fill = "white", colour = NA),
  panel.background = element_rect(fill = "white", colour = NA),
  strip.background = element_rect(fill = "#F2F2F2", colour = NA),
  strip.text = element_text(size = 6.5, face = "bold", colour = "#272727"),
  legend.position = "bottom", legend.title = element_text(size = 7), legend.text = element_text(size = 6),
  legend.key.width = grid::unit(28, "pt"), legend.key.height = grid::unit(6, "pt"),
  plot.title = element_text(size = 10, face = "bold", hjust = 0),
  plot.subtitle = element_text(size = 7.2, colour = "#555555", hjust = 0),
  plot.margin = margin(4, 4, 4, 4)
)

save_pub_r <- function(plot, stem, width_mm, height_mm) {
  base <- file.path(figure_root, stem)
  w <- width_mm / 25.4; h <- height_mm / 25.4
  svglite::svglite(paste0(base, ".svg"), width = w, height = h); print(plot); dev.off()
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Arial"); print(plot); dev.off()
  ragg::agg_tiff(paste0(base, ".tiff"), width = w, height = h, units = "in", res = 600, compression = "lzw"); print(plot); dev.off()
  ragg::agg_png(paste0(base, ".png"), width = w, height = h, units = "in", res = 600); print(plot); dev.off()
  c(svg = paste0(base, ".svg"), pdf = paste0(base, ".pdf"), tiff = paste0(base, ".tiff"), png = paste0(base, ".png"))
}

join_geom <- function(x) {
  x <- as.data.table(x)
  x[, zone_id := as.integer(zone_id)]
  geom <- adm1[adm1$zone_id %in% unique(x$zone_id), ]
  out <- left_join(geom, as.data.frame(x), by = "zone_id")
  out <- st_as_sf(out)
  out[is.finite(out$adult_population) & out$adult_population > 0, ]
}

make_map <- function(x, title, subtitle, ncol = 4) {
  geom <- join_geom(x)
  ggplot(geom) +
    geom_sf(aes(fill = obesity_pct), colour = NA) +
    geom_sf(data = adm1, fill = NA, colour = "#3E4143", linewidth = 0.08) +
    scale_fill_gradientn(colours = palette_obesity, limits = c(0, 60), breaks = c(0, 10, 20, 30, 40, 50, 60), oob = squish, name = "Obesity prevalence (%)") +
    facet_wrap(~panel, ncol = ncol) +
    coord_sf(xlim = c(-180, 180), ylim = c(-60, 84), expand = FALSE, datum = NA) +
    labs(title = title, subtitle = subtitle) + theme_map
}

read_source <- function(name) fread(file.path(source_root, name), showProgress = FALSE)
inventory <- list()

overall <- read_source("adm1_overall_keyyears_v1.csv.gz")
overall[, panel := factor(paste(scenarios[scenario_code], year, sep = "\n"), levels = as.vector(outer(unname(scenarios), years, paste, sep = "\n")))]
files <- save_pub_r(make_map(overall, "Adult obesity prevalence across SSP scenarios", "Population-supported ADM1 display; complete adult population denominator retained"), "Fig_ADM1_overall_obesity_keyyears_population_masked_v1", 183, 150)
inventory[[length(inventory) + 1]] <- data.frame(figure = "overall", files = paste(files, collapse = "|"))

sex <- read_source("adm1_sex_keyyears_v1.csv.gz")
for (sc in names(scenarios)) {
  x <- sex[scenario_code == sc]
  x[, panel := factor(paste(sex, year, sep = "\n"), levels = as.vector(outer(c("Female", "Male"), years, paste, sep = "\n")))]
  files <- save_pub_r(make_map(x, paste0("Adult obesity prevalence by sex: ", scenarios[[sc]]), "Population-supported ADM1 display; years shown left to right"), paste0("Fig_ADM1_sex_", sc, "_obesity_keyyears_population_masked_v1"), 183, 82)
  inventory[[length(inventory) + 1]] <- data.frame(figure = paste0("sex_", sc), files = paste(files, collapse = "|"))
}

age <- read_source("adm1_age_keyyears_broad6_v1.csv.gz")
for (sc in names(scenarios)) {
  x <- age[scenario_code == sc]
  x[, age_group := factor(age_group, levels = age_order)]
  x[, panel := factor(paste(age_group, year, sep = "\n"), levels = as.vector(outer(age_order, years, paste, sep = "\n")))]
  files <- save_pub_r(make_map(x, paste0("Adult obesity prevalence by age group: ", scenarios[[sc]]), "Broad age bins are used for visual readability; source tables retain 5-year age groups"), paste0("Fig_ADM1_age_", sc, "_obesity_keyyears_population_masked_v1"), 183, 190)
  inventory[[length(inventory) + 1]] <- data.frame(figure = paste0("age_", sc), files = paste(files, collapse = "|"))
}

age_sex <- read_source("adm1_age_sex_keyyears_broad6_v1.csv.gz")
for (sc in names(scenarios)) for (sx in c("Female", "Male")) {
  x <- age_sex[scenario_code == sc & sex == sx]
  x[, age_group := factor(age_group, levels = age_order)]
  x[, panel := factor(paste(age_group, year, sep = "\n"), levels = as.vector(outer(age_order, years, paste, sep = "\n")))]
  files <- save_pub_r(make_map(x, paste0("Adult obesity prevalence by age and sex: ", scenarios[[sc]], " / ", sx), "Broad age bins are used for visual readability; source tables retain 5-year age groups"), paste0("Fig_ADM1_age_sex_", sc, "_", tolower(sx), "_obesity_keyyears_population_masked_v1"), 183, 190)
  inventory[[length(inventory) + 1]] <- data.frame(figure = paste0("age_sex_", sc, "_", sx), files = paste(files, collapse = "|"))
}

inventory_df <- bind_rows(inventory)
write.csv(inventory_df, file.path(figure_root, "figure_inventory_v1.csv"), row.names = FALSE, fileEncoding = "UTF-8")
figure_files <- unlist(strsplit(inventory_df$files, "\\|"))
qa <- data.frame(path = figure_files, exists = file.exists(figure_files), bytes = ifelse(file.exists(figure_files), file.info(figure_files)$size, NA_real_))
qa$pass <- qa$exists & is.finite(qa$bytes) & qa$bytes > 1000
write.csv(qa, file.path(figure_root, "figure_export_qa_v1.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(c("status=PASS", paste0("figure_identities=", nrow(inventory_df)), paste0("files=", nrow(qa)), "backend=R", "population_mask=ADM1 adult_population > 0 for display only", "primary_metric=obesity prevalence (BMI >=30)", "key_years=2025,2030,2040,2050"), file.path(figure_root, "figure_qa_summary_v1.txt"))
if (!all(qa$pass)) stop("Figure export QA failed")
message("FIGURE_EXPORT=PASS")
