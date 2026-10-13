# Pipeline reproduction guide

Run stages in the order below from the repository root. Python scripts are
run with the conda environment (`conda activate bmi-burden`); R scripts with
`Rscript`. PowerShell wrappers (`.ps1`) orchestrate multi-step gates and are
optional conveniences — the underlying Python/R steps can be run directly.

Paths are centralized in `code/config/paths.py` (Python) and
`code/config/paths.R` (R); see README for the environment-variable overrides.

---

## s00 — Study freeze and scaffold
```
pwsh code/s00_setup/00_initialize_future_scenarios_v1.ps1
pwsh code/s00_setup/01_complete_stage00_01_inventory_v1.ps1
```
Creates the frozen study specification and the output directory scaffold.

## s01 — Input inventory
```
python code/s01_input_inventory/30_audit_crc_paf_inputs_v1.py
```
Audits that all required input files are present and readable. GBD 2021 /
GBD Foresight inputs are obtained from the GHDx portals with the query
parameters listed in docs/DATA.md and placed under `data/`.

## s02 — Covariate gates (Gate B / Gate C) and full-covariate extension
```
pwsh    code/s02_covariate_gates/run_gate_b_A4_remediated_v2.ps1   # Gate B audit of future covariates
Rscript code/s02_covariate_gates/01_build_gate_c_v1.R              # Gate C harmonized covariates
Rscript code/s02_covariate_gates/02_remediate_gate_c_country_coverage_v1.R
python  code/s02_covariate_gates/01_audit_and_build_pm25_annual_v1.py
Rscript code/s02_covariate_gates/02_build_full_covariates_0p5_v1.R
Rscript code/s02_covariate_gates/02b_build_pm25_trailing5y_0p5_v2.R
Rscript code/s02_covariate_gates/03_temporal_geometry_qa_v1.R
Rscript code/s02_covariate_gates/03b_pm25_transition_qa_v2.R
Rscript code/s02_covariate_gates/05_final_geometry_filecount_qa_v2.R
python  code/s02_covariate_gates/06_finalize_full_covariate_gate_v2.py
```

## s03 — National BMI anchors (2025–2050)
```
python code/s03_national_bmi_anchors/05_audit_and_build_national_bmi7_anchors_v2.py
python code/s03_national_bmi_anchors/06_independent_qa_national_bmi7_anchors_v1.py
python code/s03_national_bmi_anchors/07_finalize_phase3_national_anchors_v1.py
```

## s04 — Future-capable MRP model
```
pwsh    code/s04_mrp_model/run_phase4b_d1_fits_v1.ps1              # D1 16-covariate fits (Stan)
python  code/s04_mrp_model/evaluate_phase4b_d1_noninferiority_v1.py
python  code/s04_mrp_model/evaluate_phase4b_d1_stabilized_v1.py
Rscript code/s04_mrp_model/01_rebuild_training_pm25_mmr_v1.R       # full-covariate training table
Rscript code/s04_mrp_model/pathfinder_prep_data_v1.R
Rscript code/s04_mrp_model/mrp_full_covariate_refit_validation_v1.R
Rscript code/s04_mrp_model/pathfinder_analyze_v1.R
python  code/s04_mrp_model/evaluate_full_covariate_temporal_backtest_v1.py
python  code/s04_mrp_model/evaluate_full_covariate_source_holdout_v1.py
python  code/s04_mrp_model/build_full_covariate_noninferiority_gate_v1.py
```
The Stan program is `code/s04_mrp_model/bmi7_mrp_cmdstan.stan`.
This stage requires CmdStan (`cmdstanr::install_cmdstan()`).

## s05 — Historical BMI layer (2015–2024) and temporal backtesting
```
python  code/s05_historical_bmi/prepare_phase5_temporal_design_v1.py
Rscript code/s05_historical_bmi/phase5_baseline_temporal_task_v1.R
Rscript code/s05_historical_bmi/phase5_candidate_temporal_task_v1.R
python  code/s05_historical_bmi/evaluate_phase5_temporal_backtesting_v1.py
Rscript code/s05_historical_bmi/run_historical_25plus_compat_v1.R
Rscript code/s05_historical_bmi/check_historical2024_future2025_continuity_v1.R
```

## s06 — Future BMI production (4 SSPs × 2025–2050)
```
Rscript code/s06_future_bmi_production/run_full_covariate_annual_0p5deg_production_v1.R
python  code/s06_future_bmi_production/01_run_full_covariate_annual_0p1deg_batch_v1.py
        # dispatches per-scenario-year tasks built from the four templates
        # (01_harmonize/02_model/03_rake/04_qa_task_template)
python  code/s06_future_bmi_production/phase7_multiscale_delivery_v3.py
python  code/s06_future_bmi_production/07_qa_adm1_v1.py
python  code/s06_future_bmi_production/08_prepare_figure_source_data_v1.py
Rscript code/s06_future_bmi_production/09_plot_adm1_population_masked_atlas_v1.R
python  code/s06_future_bmi_production/01_rake_ssp245_2025_bridge_v1.py   # bridge sensitivity
python  code/s06_future_bmi_production/03_run_ssp245_keyyears_bridge_v1.py
python  code/s06_future_bmi_production/04_qa_ssp245_keyyears_bridge_v1.py
python  code/s06_future_bmi_production/05_derive_bridge_aggregates_v2.py
```

## s07 — RR freeze (data-only)
No code: the WCRF/AICR 2024 dose–response table for colorectal cancer
(7 BMI categories × 2 sexes, reference BMI 22.5 kg/m²) is transcribed as
described in docs/DATA.md. The 65-disease RR crosswalk is built in s11.

## s08 — CRC population-attributable fractions
```
python code/s08_crc_paf/04_run_crc_paf_full_controller_v1.py   # controller
#     → dispatches 02_crc_paf_full_task_v1.py (104 scenario-year tasks)
python code/s08_crc_paf/03_finalize_crc_paf_full_v1.py
```

## s09 — CRC attributable burden
```
python code/s09_crc_burden/01_run_crc_attributable_burden_v1.py
```

## s10 — Historical benchmark against GBD 2021
```
python code/s10_gbd_benchmark/01_run_crc_historical_benchmark_v3.py
python code/s10_gbd_benchmark/02_plot_crc_historical_benchmark_v3.py
```

## s11 — All-disease extension (65 causes)
```
python code/s11_all_disease_extension/01_harmonize_gbd_inputs_v1.py
python code/s11_all_disease_extension/02_build_rr_crosswalk_v1.py
python code/s11_all_disease_extension/03_compute_paf_all_causes_v1.py     # national PAF
python code/s11_all_disease_extension/01_compute_grid_paf_v1.py           # 0.1° grid PAF
python code/s11_all_disease_extension/04_burden_and_benchmark_v1.py
python code/s11_all_disease_extension/01_validate_pop_times_paf_v1.py     # spatial validation
python code/s11_all_disease_extension/01_compute_paf_weights_all_causes_v1.py
python code/s11_all_disease_extension/02_allocate_burden_density_v1.py    # 0.1° burden density
python code/s11_all_disease_extension/01_compute_burden_density_2015_v1.py
python code/s11_all_disease_extension/03_cause_level_2015_deaths_v1.py
python code/s11_all_disease_extension/04_cause_level_2015_dalys_v1.py
```

## s12 — Figures

| Figure | Script(s) in code/s12_figures/ |
|---|---|
| Fig. 2 (BMI trajectories + spatial) | 39_plot_figure2_bmi_global_2015_2050_v1.py, 40_aggregate_sex_trend_figure2_v1.py, 41_plot_figure2_panels_ABCDEF_v1.py |
| Fig. 3 (all-disease burden) | 66_plot_figure3A_alldisease_burden_trends_v1.py, 67_plot_figure3A_disease_composition_trends_v1.py, 68_plot_figure3A_stacked_burden_by_disease_v1.py, 71_plot_figure3BC_grid_maps_v1.py, 72_plot_figure3H_top3_cancers_panel_v1.py, 73_plot_figure3H_top2_disease_maps_v1.py |
| Fig. 4 (velocity) | 75_plot_figure4_velocity_real_v1.py, 76_plot_figure4_dalys_maps_v1.py |
| Fig. 5 (multi-disease burden) | 54_plot_figure5_multidisease_burden_v1.py |
| Fig. S2/S3 (BMI fill) | 50_plot_supplementary_s2_s3_obesity_fill_v1.py |
| Fig. S4 (four-SSP BMI-7 atlas) | 51_plot_bmi7_four_ssp_7x4_1800dpi_v1.py |
| Fig. S7 (multi-disease PAF) | 55_plot_supplementary_s7_multidisease_paf_v1.py |
| Supplementary benchmark | 53_plot_supplementary_multidisease_benchmark_v5.py |
| Fig. S15–S18 (per-disease composites) | 79_plot_figureS15_composites_v2.py (uses 81_build_crc_2015_density_v1.py once to derive the CRC 2015 grid) |
| BMI-7 grid validation | 76_plot_validation_bmi7_grid_v1.py |

`08_plot_figure1B_historical_template_test_v2.py` and
`audit_panel_alignment.py` are shared helper modules imported by the figure
scripts; they are not run directly.
