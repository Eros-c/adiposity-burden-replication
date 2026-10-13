# External data sources

All inputs are public datasets. Download them and place them under `data/`
with the directory names below (scripts build paths as
`_cfg$DATA / "<dirname>" / ...`). Total download size ≈ 60–80 GB.

| Directory under `data/` | Source | Content | Access |
|---|---|---|---|
| `GBD2021_downloads/results_total_burden/` | GBD 2021 Results (ghdx.healthdata.org) | Total deaths & DALYs, 65 causes × 204 countries, 2015–2021, both sexes, Number | GHDx Results tool (free registration); query parameters in `GBD数据获取参数清单_多疾病_v1.md` (project notes) |
| `GBD2021_downloads/results_high_bmi/` | GBD 2021 Results | High-BMI-attributable deaths & DALYs, 39 risk-linked causes, 2015–2021 | same |
| `GBD2021_downloads/foresight/` | GBD Foresight (vizhub.healthdata.org/gbd-foresight) | Reference-scenario total burden, 2022–2050 | web export |
| `GBD2021_foresight_exports/` | GBD Foresight | CRC reference burden 2024–2050 (`foresight_crc_reference_2024_2050.csv`) | web export |
| `Adult_BMI_forecast_1990_2050/` | GBD 2021 Adult Overweight/Obesity forecast (GHDX, released 2025-03) | National adult overweight/obesity prevalence, 1990–2050 | GHDX record |
| `NCD_RisC/` | NCD Risk Factor Collaboration | National BMI distribution (7-category), by age/sex/year | ncd-risc.org |
| `SSP/` | IIASA SSP Basic Drivers v3.2; WiC SSP V14 | National & gridded population, GDP, age-sex structure | sspdb / iiasa.ac.at |
| `ISIMIP/` | ISIMIP | 0.25°–0.5° gridded SSP population | isimip.org |
| `WorldPop/` | WorldPop | 2025 age-sex 0.1° population | worldpop.org |
| `GDP_gridded/` | Wang & Sun 2022 (Nature Sci Data) | 0.25° gridded GDP, calibrated to IIASA/OECD ENV-Growth 2025 | doi:10.1038/s41597-022-01300-x |
| `CMIP6/` | CMIP6 (7 models: ACCESS-CM2, CanESM5, CESM2-WACCM, FGOALS-g3, MIROC6, MPI-ESM1-2-HR, MRI-ESM2-0) | tas, hurs, PM2.5 (`mmrpm2p5`, NorESM2-LM) × 4 SSPs | esgf-node.llnl.gov |
| `ERA5/` | ECMWF ERA5 | 2 m temperature / humidity, bias-correction reference | cds.climate.copernicus.eu |
| `GHS/` | GHSL | GHS-SMOD urbanization, GHS built-up surface | jeodpp.jrc.ec.europa.eu |
| `elevation/`, `accessibility/` | SRTM / JRC accessibility | Static spatial covariates | public |
| `ne_10m_admin_0_countries.zip` | Natural Earth 10 m | Country boundaries (also ADM1) | naturalearthdata.com |
| `WCRF/` | WCRF/AICR 2024 report | CRC dose–response by BMI category (primary RR table) | wcrf.org |

## Notes

- GBD downloads are obtained from the GHDx portals (Results tool and
  Foresight visualization) with the query parameters above and placed under
  `data/` as documented.
- All outputs of the pipeline (national tables, 0.5°/0.1°/ADM1 rasters,
  figures) are regenerated under `results/` and are not distributed with
  this repository.
- The CRC 2015 0.1° density grid used by the per-disease composite figures
  is derived by `code/s12_figures/81_build_crc_2015_density_v1.py` from the
  s11 weight rasters and the historical benchmark table.
