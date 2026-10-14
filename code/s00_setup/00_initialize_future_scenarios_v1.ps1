if (-not $env:BMI_ROOT) { $env:BMI_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path }  # auto-detect repo root; override with $env:BMI_ROOT

[CmdletBinding()]
param(
    [string]$OutputRoot = '$env:BMI_ROOT\results\未来情景预测'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Utf8BomCsv {
    param([Parameter(Mandatory)]$Data, [Parameter(Mandatory)][string]$Path)
    $text = $Data | ConvertTo-Csv -NoTypeInformation
    [System.IO.File]::WriteAllLines($Path, $text, [System.Text.UTF8Encoding]::new($true))
}

function Write-Utf8NoBom {
    param([Parameter(Mandatory)][string]$Text, [Parameter(Mandatory)][string]$Path)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Assert-NewFile {
    param([Parameter(Mandatory)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing artifact: $Path"
    }
}

$stageDirs = @(
    '00_plan_manifest',
    '01_input_inventory',
    '02_harmonized_covariates',
    '03_national_bmi_anchors',
    '04_future_capable_mrp',
    '05_historical_bmi_1990_2024',
    '06_future_bmi_2025_2050',
    '07_rr_cause_crosswalk',
    '08_colorectal_cancer_paf',
    '09_colorectal_cancer_burden',
    '10_gbd_comparison',
    '11_all_disease_extension',
    '12_figures',
    '13_tables',
    '14_uncertainty',
    '15_code_logs_checksums',
    '15_code_logs_checksums\logs',
    '15_code_logs_checksums\scripts'
)
foreach ($dir in $stageDirs) {
    New-Item -ItemType Directory -Path (Join-Path $OutputRoot $dir) -Force | Out-Null
}

$paths = [ordered]@{
    study_spec = Join-Path $OutputRoot '00_plan_manifest\frozen_study_spec_v1.json'
    stage_manifest = Join-Path $OutputRoot '00_plan_manifest\stage_gate_manifest_v1.csv'
    baseline_roots = Join-Path $OutputRoot '00_plan_manifest\baseline_roots_v1.csv'
    baseline_files = Join-Path $OutputRoot '00_plan_manifest\baseline_artifact_manifest_v1.csv'
    baseline_summary = Join-Path $OutputRoot '00_plan_manifest\baseline_summary_v1.json'
    stage00_report = Join-Path $OutputRoot '00_plan_manifest\stage00_freeze_report_CN_v1.md'
    inventory = Join-Path $OutputRoot '01_input_inventory\input_inventory_v1.csv'
    contract = Join-Path $OutputRoot '01_input_inventory\future_data_contract_v1.csv'
    gaps = Join-Path $OutputRoot '01_input_inventory\blocking_data_gaps_v1.csv'
    sources = Join-Path $OutputRoot '01_input_inventory\authoritative_download_sources_v1.csv'
    readiness = Join-Path $OutputRoot '01_input_inventory\local_data_readiness_v1.json'
    stage01_report = Join-Path $OutputRoot '01_input_inventory\stage01_inventory_report_CN_v1.md'
    workflow = Join-Path $OutputRoot '00_plan_manifest\implementation_workflow_CN_v1.md'
    sha = Join-Path $OutputRoot '15_code_logs_checksums\SHA256SUMS_stage00_01_v1.csv'
}
foreach ($path in $paths.Values) { Assert-NewFile $path }

$studySpec = [ordered]@{
    specification_version = 'future_scenarios_v1'
    frozen_at = (Get-Date).ToString('o')
    population = [ordered]@{
        primary = 'Adults aged 25 years and older'
        supplementary = 'Adults aged 20-24 years'
    }
    periods = [ordered]@{
        historical = '1990-2024'
        future = '2025-2050'
        annual_computation = $true
        key_years = @(1990, 2000, 2010, 2021, 2024, 2030, 2040, 2050)
    }
    scenarios = [ordered]@{
        primary = 'SSP2-4.5'
        sensitivity = @('SSP1-2.6', 'SSP3-7.0', 'SSP5-8.5')
    }
    spatial_scales = [ordered]@{
        primary_grid = '0.5 degree'
        administrative = 'ADM1'
        supplementary_grid = '0.1 degree'
        excluded_from_future_inference = '1 km'
    }
    bmi_model = [ordered]@{
        primary = 'Refitted future-capable Bayesian multinomial logistic-normal MRP'
        historical_baseline = 'NCD-RisC country-year-age-sex seven-category BMI composition'
        old_model_role = 'Read-only comparison baseline; not a direct 2050 predictor'
    }
    prototype_disease = 'Colorectal cancer'
    primary_disease_output = 'Disease-specific PAF'
    secondary_disease_outputs = @('Attributable incidence', 'Attributable deaths', 'Attributable DALYs')
    inference_constraints = @(
        'Future SSP2 BMI anchors constrained to GBD overweight and obesity forecasts',
        'GBD-constrained quantities are not independent validation',
        'No batch all-disease mapping before colorectal cancer prototype passes all gates'
    )
}
Write-Utf8NoBom -Text ($studySpec | ConvertTo-Json -Depth 8) -Path $paths.study_spec

$stageManifest = @(
    [pscustomobject]@{stage=0;name='冻结现有成果';status='completed';gate='基线可恢复且哈希完整';next_action='none'},
    [pscustomobject]@{stage=1;name='补充关键数据';status='blocked_partial';gate='所有关键未来数据满足数据合同';next_action='获取GBD未来BMI、CRC负担及SSP/CMIP6机器可读数据'},
    [pscustomobject]@{stage=2;name='统一空间与时间数据';status='not_started';gate='第1步通过';next_action='wait'},
    [pscustomobject]@{stage=3;name='建立国家七分类BMI基准';status='not_started';gate='第2步通过';next_action='wait'},
    [pscustomobject]@{stage=4;name='重新拟合未来可用MRP';status='not_started';gate='第3步通过且非劣性标准冻结';next_action='wait'},
    [pscustomobject]@{stage=5;name='时间外推验证';status='not_started';gate='第4步模型可运行';next_action='wait'},
    [pscustomobject]@{stage=6;name='生成1990-2050年BMI结果';status='not_started';gate='第4和5步均通过';next_action='wait'},
    [pscustomobject]@{stage=7;name='冻结疾病效应和GBD病因映射';status='not_started';gate='CRC参数唯一且病因一对一';next_action='wait'},
    [pscustomobject]@{stage=8;name='计算结直肠癌PAF';status='not_started';gate='BMI组成与RR参数QA通过';next_action='wait'},
    [pscustomobject]@{stage=9;name='计算结直肠癌归因负担';status='not_started';gate='CRC GBD总负担可用';next_action='wait'},
    [pscustomobject]@{stage=10;name='不确定性传播';status='not_started';gate='全部不确定性输入可抽样';next_action='wait'},
    [pscustomobject]@{stage=11;name='国家层面比较';status='not_started';gate='约束与独立比较口径分开';next_action='wait'},
    [pscustomobject]@{stage=12;name='扩展至全部疾病';status='not_started';gate='CRC样板全部通过';next_action='wait'}
)
Write-Utf8BomCsv -Data $stageManifest -Path $paths.stage_manifest

$baselineRoots = @(
    [pscustomobject]@{
        root_id='regional_steps_enhanced_mrp_20260812'
        path='$env:BMI_ROOT\results\降尺度\experiments\BMI7_MRP_STEPS_regional_augmentation_20260812'
        role='Current enhanced MRP, validation, 2015-2024 three-scale mapping and QA'
        frozen='true'
    },
    [pscustomobject]@{
        root_id='pre_steps_enhanced_mrp_stage09'
        path='$env:BMI_ROOT\results\降尺度\experiments\BMI7_model_comparison_20260728\gated_run_v1\stage_09_conditional_mapping_enhanced_mrp_v1'
        role='Earlier enhanced MRP mapping baseline retained for provenance'
        frozen='true'
    }
)
Write-Utf8BomCsv -Data $baselineRoots -Path $paths.baseline_roots

$baselineFileRows = [System.Collections.Generic.List[object]]::new()
foreach ($root in $baselineRoots) {
    if (-not (Test-Path -LiteralPath $root.path)) {
        throw "Frozen baseline root missing: $($root.path)"
    }
    $files = @(Get-ChildItem -LiteralPath $root.path -Recurse -File -ErrorAction Stop)
    foreach ($file in $files) {
        $relative = [System.IO.Path]::GetRelativePath($root.path, $file.FullName)
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $baselineFileRows.Add([pscustomobject]@{
            root_id = $root.root_id
            relative_path = $relative
            full_path = $file.FullName
            size_bytes = [int64]$file.Length
            last_write_utc = $file.LastWriteTimeUtc.ToString('o')
            sha256 = $hash
        })
    }
}
Write-Utf8BomCsv -Data $baselineFileRows -Path $paths.baseline_files

$baselineSummary = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    frozen_roots = $baselineRoots.Count
    frozen_files = $baselineFileRows.Count
    frozen_bytes = [int64](($baselineFileRows | Measure-Object -Property size_bytes -Sum).Sum)
    no_files_copied = $true
    no_existing_files_modified = $true
    current_model = [ordered]@{
        family = 'Bayesian multinomial logistic-normal MRP'
        training_rows = 53777
        training_countries = 56
        source_families = 9
        fixed_effect_features = 22
        inference = 'Stan mean-field variational inference, 15000 iterations, 1000 approximate posterior draws'
        mapping_years = '2015-2024'
        mapping_scales = @('0.1 degree', '0.5 degree', 'ADM1')
        primary_validation = [ordered]@{
            countries = 20
            minimum_effective_n = 20
            internal_weighted_r2 = 0.8597087008884492
            internal_w2 = 1.7652638431544
            internal_obesity_mae_pp = 2.591713939284785
            country_holdout_weighted_r2 = 0.787747210672958
            country_holdout_w2 = 1.9631089982898118
            country_holdout_obesity_mae_pp = 3.3472117276282694
            country_holdout_within_anchor_r = 0.3351130796173073
        }
        future_use = 'comparison_baseline_only'
    }
}
Write-Utf8NoBom -Text ($baselineSummary | ConvertTo-Json -Depth 8) -Path $paths.baseline_summary

$inventoryDefinitions = @(
    [ordered]@{id='baseline_mrp';dataset='区域STEPS增强MRP与2015-2024 BMI7结果';local_path=$baselineRoots[0].path;required_stage='0';status='available';years='2015-2024';spatial='0.1°, 0.5°, ADM1';scenario='historical';age_sex='age × sex';unit='BMI category proportions';notes='Frozen comparison baseline'},
    [ordered]@{id='ncd_risc_bmi7';dataset='NCD-RisC国家-年份-年龄-性别BMI组成';local_path='$env:BMI_ROOT\data\肥胖\全球\NCD_RisC\国家-年份-年龄-性别肥胖';required_stage='3';status='available';years='1980-2024';spatial='country';scenario='historical';age_sex='age × sex';unit='proportion';notes='Historical seven-category anchor'},
    [ordered]@{id='gbd_adult_bmi_forecast';dataset='GBD 2021成人超重和肥胖预测机器可读表';local_path='';required_stage='3';status='missing_blocking';years='2022-2050';spatial='country';scenario='GBD reference';age_sex='25+ age × sex';unit='prevalence with 95% UI';notes='Paper located; machine-readable estimates not present locally'},
    [ordered]@{id='gbd_crc_total_forecast';dataset='GBD结直肠癌总发病、死亡和DALY历史及预测';local_path='';required_stage='9';status='missing_blocking';years='1990-2050';spatial='country';scenario='GBD reference/forecast';age_sex='age × sex';unit='counts, rates and 95% UI';notes='Cause-specific detailed extract required'},
    [ordered]@{id='gbd_crc_high_bmi';dataset='GBD结直肠癌高BMI归因发病、死亡和DALY';local_path='';required_stage='10-11';status='missing_blocking';years='historical and forecast where available';spatial='country';scenario='GBD';age_sex='age × sex';unit='attributable counts/rates and 95% UI';notes='Needed for comparison, not for independent validation if used as constraint'},
    [ordered]@{id='local_gbd_2020';dataset='现有本地GBD高BMI聚合结果';local_path='$env:BMI_ROOT\data\GBD数据';required_stage='audit';status='available_not_sufficient';years='2020 only';spatial='country';scenario='historical';age_sex='aggregated';unit='attributable burden';notes='Three aggregate cause files; cannot support 1990-2050 CRC pipeline'},
    [ordered]@{id='ssp_population_grid';dataset='SSP网格人口';local_path='';required_stage='2,4,6';status='missing_blocking';years='2010-2100 decadal';spatial='1/8° or source native';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='total/urban-rural; age-sex if obtainable';unit='persons';notes='Must be harmonized to 0.1° then aggregated'},
    [ordered]@{id='ssp_population_country';dataset='SSP国家人口、城乡和年龄性别预测';local_path='';required_stage='2,3,6';status='missing_blocking';years='2020-2050';spatial='country';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='age × sex';unit='persons or shares';notes='IIASA SSP database'},
    [ordered]@{id='ssp_gdp_grid';dataset='SSP网格GDP或人均GDP';local_path='';required_stage='2,4,6';status='missing_blocking';years='2030-2050';spatial='0.25° preferred';scenario='SSP1,SSP2,SSP3,SSP5';age_sex='none';unit='PPP currency per capita';notes='Bridge from 2024 historical GDP by ratio'},
    [ordered]@{id='cmip6_climate';dataset='NEX-GDDP-CMIP6温度和湿度';local_path='';required_stage='2,4,6';status='missing_blocking';years='2015-2050';spatial='0.25°';scenario='SSP126,SSP245,SSP370,SSP585';age_sex='none';unit='tas K/°C; hurs % or derivable humidity';notes='At least five GCMs; bias-correct against ERA5'},
    [ordered]@{id='worldpop_historical';dataset='WorldPop年龄性别人口网格';local_path='$env:BMI_ROOT\data\人口结构网格数据';required_stage='2,5,6';status='available_partial';years='2006,2012-2025';spatial='1 km';scenario='historical';age_sex='age × sex';unit='persons';notes='Does not cover full 1990-2005 historical period'},
    [ordered]@{id='historical_gdp_grid';dataset='Kummu等历史人均GDP网格';local_path='$env:BMI_ROOT\data\GDP网格数据';required_stage='2,4,5';status='available';years='1990-2024';spatial='5 arc-min';scenario='historical';age_sex='none';unit='PPP per capita';notes='Historical bridge source'},
    [ordered]@{id='historical_pm25';dataset='ACAG SatPM2.5';local_path='$env:BMI_ROOT\data\PM2.5网格数据';required_stage='sensitivity';status='available_sensitivity_only';years='historical';spatial='source native';scenario='historical';age_sex='none';unit='µg/m³';notes='Excluded from future main model'},
    [ordered]@{id='era5_temperature';dataset='ERA5 2m temperature';local_path='$env:BMI_ROOT\data\协变量数据\ERA5：温度';required_stage='2,4,5';status='available';years='historical';spatial='source native';scenario='historical';age_sex='none';unit='K/°C';notes='CMIP6 bias correction reference'},
    [ordered]@{id='era5_humidity';dataset='ERA5 2m temperature and dewpoint humidity source';local_path='$env:BMI_ROOT\data\协变量数据\ERA5：湿度';required_stage='2,4,5';status='available';years='historical';spatial='source native';scenario='historical';age_sex='none';unit='K and derived RH %';notes='CMIP6 bias correction reference'},
    [ordered]@{id='ghs_smod';dataset='GHS-SMOD城市化分类';local_path='$env:BMI_ROOT\data\城市化网格数据_GHS_SMOD';required_stage='2,4,5';status='available_partial';years='2005,2010,2015,2020';spatial='1 km';scenario='historical';age_sex='none';unit='settlement class';notes='Historical urbanization; future requires SSP-compatible source'},
    [ordered]@{id='rr_workbook';dataset='BMI-疾病RR资料表';local_path='$env:BMI_ROOT\data\肥胖-疾病剂量反应关系\BMI_疾病关联.xlsx';required_stage='7-8';status='available_requires_crc_freeze';years='literature';spatial='none';scenario='none';age_sex='depends on study';unit='RR per 5 kg/m²';notes='CRC row must be selected and evidence/outcome aligned'},
    [ordered]@{id='static_elevation';dataset='GMTED2010高程';local_path='$env:BMI_ROOT\data\协变量数据\GMTED2010：高程数据';required_stage='2,4,6';status='check_path';years='static';spatial='source native';scenario='static';age_sex='none';unit='metres';notes='Static future predictor'},
    [ordered]@{id='static_accessibility';dataset='2015城市和港口交通时间';local_path='$env:BMI_ROOT\data\Travel time to cities and ports in the year 2015';required_stage='2,4,6';status='check_path';years='2015 static';spatial='~1 km';scenario='static';age_sex='none';unit='minutes';notes='Static future predictor'}
)

$inventoryRows = foreach ($entry in $inventoryDefinitions) {
    $exists = if ([string]::IsNullOrWhiteSpace($entry.local_path)) { $false } else { Test-Path -LiteralPath $entry.local_path }
    $files = if ($exists) { @(Get-ChildItem -LiteralPath $entry.local_path -Recurse -File -ErrorAction SilentlyContinue) } else { @() }
    [pscustomobject]@{
        dataset_id = $entry.id
        dataset = $entry.dataset
        required_stage = $entry.required_stage
        status = if (($entry.status -eq 'check_path') -and $exists) { 'available' } elseif (($entry.status -eq 'check_path') -and -not $exists) { 'missing_blocking' } else { $entry.status }
        local_path = $entry.local_path
        path_exists = $exists
        file_count = $files.Count
        size_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
        years = $entry.years
        spatial_resolution = $entry.spatial
        scenario_coverage = $entry.scenario
        age_sex_coverage = $entry.age_sex
        unit = $entry.unit
        notes = $entry.notes
    }
}
Write-Utf8BomCsv -Data $inventoryRows -Path $paths.inventory

$contractRows = @(
    [pscustomobject]@{dataset_id='gbd_adult_bmi_forecast';required_fields='location_id, location_name, iso3, year, age_group_id/name, sex_id/name, metric, overweight_mean/lower/upper, obesity_mean/lower/upper';required_years='2022-2050 annual or reproducibly interpolable';required_scenarios='GBD reference';required_ui='yes';acceptance='25+; mutually consistent overweight and obesity definitions; stable IDs'},
    [pscustomobject]@{dataset_id='gbd_crc_total_forecast';required_fields='cause_id, location_id, year, age_group_id, sex_id, measure_id, metric_id, mean, lower, upper';required_years='1990-2050';required_scenarios='historical and forecast';required_ui='yes';acceptance='CRC cause maps one-to-one; incidence/death/DALY counts and rates separable'},
    [pscustomobject]@{dataset_id='gbd_crc_high_bmi';required_fields='cause_id, rei_id, location_id, year, age_group_id, sex_id, measure_id, metric_id, mean, lower, upper';required_years='historical and forecast where available';required_scenarios='GBD';required_ui='yes';acceptance='rei=high BMI; no parent-child cause double counting'},
    [pscustomobject]@{dataset_id='ssp_population_grid';required_fields='scenario, year, lon/lat or georeference, total population, urban/rural where available';required_years='2020-2050 at least decadal';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='preferred';acceptance='global coverage; documented population totals and no negative cells'},
    [pscustomobject]@{dataset_id='ssp_population_country';required_fields='iso3/location, scenario, year, age, sex, population, urbanization';required_years='2020-2050';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='preferred';acceptance='age-sex categories can be crosswalked to MRP poststrata'},
    [pscustomobject]@{dataset_id='ssp_gdp_grid';required_fields='scenario, year, lon/lat or georeference, GDP or GDP per capita PPP, base year/currency';required_years='2030-2050';required_scenarios='SSP1,SSP2,SSP3,SSP5';required_ui='no';acceptance='0.25° preferred; global; PPP base documented'},
    [pscustomobject]@{dataset_id='cmip6_climate';required_fields='GCM, scenario, date/year, tas, hurs or humidity inputs, georeference';required_years='2015-2050 daily or monthly';required_scenarios='SSP126,SSP245,SSP370,SSP585';required_ui='ensemble spread';acceptance='at least 5 GCMs; calendar and units documented; no mixed historical/scenario discontinuity'}
)
Write-Utf8BomCsv -Data $contractRows -Path $paths.contract

$gapRows = @(
    [pscustomobject]@{priority=1;dataset_id='gbd_adult_bmi_forecast';blocking_stage='3';owner='user/IHME access';problem='Machine-readable 2022-2050 country-age-sex overweight and obesity forecasts absent';resolution='Download source estimates with 95% UI; do not digitize figures';status='open'},
    [pscustomobject]@{priority=2;dataset_id='gbd_crc_total_forecast';blocking_stage='9';owner='user/IHME access';problem='Detailed CRC incidence, deaths and DALYs absent';resolution='Export country-year-age-sex counts/rates and UI for historical and forecast periods';status='open'},
    [pscustomobject]@{priority=3;dataset_id='gbd_crc_high_bmi';blocking_stage='10-11';owner='user/IHME access';problem='High-BMI-attributable CRC result absent';resolution='Export matching cause/rei/measure data for comparison';status='open'},
    [pscustomobject]@{priority=4;dataset_id='ssp_population_grid';blocking_stage='2';owner='download';problem='No local SSP population grids';resolution='Acquire SSP1/2/3/5 through 2050 and document source version';status='open'},
    [pscustomobject]@{priority=5;dataset_id='ssp_population_country';blocking_stage='2-3';owner='download';problem='No local SSP age-sex and urbanization projections';resolution='Acquire IIASA country projections and freeze version';status='open'},
    [pscustomobject]@{priority=6;dataset_id='ssp_gdp_grid';blocking_stage='2';owner='download';problem='No local future SSP GDP grid';resolution='Acquire 0.25° SSP GDP 2030-2050';status='open'},
    [pscustomobject]@{priority=7;dataset_id='cmip6_climate';blocking_stage='2';owner='download';problem='No local CMIP6 future temperature/humidity ensemble';resolution='Acquire at least 5 GCMs for four scenarios, 2015-2050';status='open'},
    [pscustomobject]@{priority=8;dataset_id='worldpop_historical';blocking_stage='5-6';owner='method decision';problem='Historical age-sex grid does not cover 1990-2005';resolution='Select defensible historical population backcast source or limit spatial reconstruction period';status='open'}
)
Write-Utf8BomCsv -Data $gapRows -Path $paths.gaps

$sourceRows = @(
    [pscustomobject]@{dataset_id='gbd_adult_bmi_forecast';title='Global prevalence of adult overweight and obesity, 1990-2021, with forecasts to 2050';url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11920007/';access='article/open';download_note='Machine-readable estimates may require article supplement or IHME result tool'},
    [pscustomobject]@{dataset_id='ssp_population_country';title='IIASA Shared Socioeconomic Pathways database';url='https://iiasa.ac.at/models-tools-data/ssp';access='registration may be required';download_note='Freeze database release and scenario version'},
    [pscustomobject]@{dataset_id='ssp_population_grid';title='Global population projection grids based on SSPs';url='https://data.nasa.gov/dataset/global-population-projection-grids-based-on-shared-socioeconomic-pathways-ssps-2010-2100';access='open';download_note='Acquire SSP1, SSP2, SSP3 and SSP5 layers'},
    [pscustomobject]@{dataset_id='ssp_gdp_grid';title='Global gridded GDP projections under SSPs';url='https://www.nature.com/articles/s41597-022-01300-x';access='open';download_note='Use associated repository; prefer 0.25 degree product'},
    [pscustomobject]@{dataset_id='ssp_gdp_grid';title='Gridded GDP dataset repository';url='https://doi.org/10.5281/zenodo.7898409';access='open';download_note='Record exact file hashes and units'},
    [pscustomobject]@{dataset_id='cmip6_climate';title='NASA NEX-GDDP-CMIP6';url='https://www.nasa.gov/nasa-earth-exchange-nex/gddp/downscaled-climate-projections-nex-gddp-cmip6/';access='open cloud/object store';download_note='Select at least 5 GCMs and the four frozen scenarios'},
    [pscustomobject]@{dataset_id='ghs_wup';title='GHS-WUP 2025';url='https://human-settlement.emergency.copernicus.eu/ghs_wup2025.php';access='open';download_note='Use for historical/future consistency checks, not as SSP age-sex substitute'}
)
Write-Utf8BomCsv -Data $sourceRows -Path $paths.sources

$readiness = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    stage0 = 'completed'
    stage1 = 'blocked_partial'
    datasets_total = $inventoryRows.Count
    available_or_partial = @($inventoryRows | Where-Object status -Match '^available').Count
    missing_blocking = @($inventoryRows | Where-Object status -eq 'missing_blocking').Count
    can_start_stage2 = $false
    can_fit_future_mrp = $false
    can_generate_2050_bmi = $false
    can_compute_crc_paf_prototype = $false
    can_compute_crc_burden = $false
    reason = 'Future BMI anchors, SSP population/GDP, CMIP6 climate and detailed colorectal cancer GBD extracts are absent.'
}
Write-Utf8NoBom -Text ($readiness | ConvertTo-Json -Depth 5) -Path $paths.readiness

$stage00Report = @"
# 第0步：冻结现有成果

## 结论

第0步已完成。现有区域STEPS增强MRP、验证、2015–2024年0.1°/0.5°/ADM1结果以及更早的Stage 9基线均保持只读，未复制、未覆盖、未重新拟合。

## 冻结方式

- 原始成果继续保存在原目录。
- `baseline_artifact_manifest_v1.csv`记录每个文件的完整路径、相对路径、大小、修改时间和SHA256。
- `baseline_summary_v1.json`记录当前模型、验证指标和制图范围。
- 旧模型的角色冻结为“比较基线”，不能直接作为2050年预测器。

## 当前正式基线

- 模型：贝叶斯多项Logistic-normal MRP。
- 增强训练表：53,777条记录、56个国家、9类数据来源。
- 现有制图：2015–2024年，0.1°、0.5°和ADM1。
- 20国稳定聚合验证（有效样本量≥20）：内部加权R² 0.860，国家留出加权R² 0.788。
- 国家留出中，国家锚点内肥胖偏差相关为0.335，说明国家内部空间迁移仍有明显不确定性。

## 边界

冻结不等于认可旧模型可直接外推至2050年。未来主模型必须移除无法合理预测至2050年的主效应变量，并通过时间回测和非劣性门槛。
"@
Write-Utf8NoBom -Text $stage00Report -Path $paths.stage00_report

$stage01Report = @"
# 第1步：关键数据审计

## 当前状态

第1步仅部分完成，数据门未通过。现有历史BMI、历史GDP、ERA5、WorldPop、GHS-SMOD和RR资料可以支持方法准备，但不足以运行1990–2050年正式预测。

## 关键阻断项

1. 缺少GBD 2022–2050成人超重和肥胖的国家×年份×年龄×性别机器可读预测及95% UI。
2. 缺少结直肠癌国家×年份×年龄×性别总发病、死亡和DALY历史及未来预测。
3. 缺少结直肠癌高BMI归因结果，无法进行未被约束指标的GBD比较。
4. 缺少SSP1/2/3/5人口网格、国家年龄性别人口与城市化预测。
5. 缺少SSP网格GDP和四情景CMIP6温湿度集合。
6. 现有WorldPop年龄性别网格不能覆盖1990–2005年，需要补充历史回推来源或调整历史空间重建起点。

## 已核实但不够用的数据

`$env:BMI_ROOT\data\GBD数据`只有2020年聚合高BMI归因结果，不能替代结直肠癌1990–2050年年龄性别详细数据。

## 数据门

在上述阻断项完成前：

- 不拟合未来MRP；
- 不生成2050年BMI地图；
- 不计算正式结直肠癌PAF或疾病负担；
- 不扩展到全部疾病。

下载后应先按`future_data_contract_v1.csv`核验字段、单位、年份、年龄、性别、情景和不确定性，再进入第2步。
"@
Write-Utf8NoBom -Text $stage01Report -Path $paths.stage01_report

$workflow = @"
# 全球BMI与疾病负担未来情景预测：执行流程

```text
冻结历史基线
    ↓
补齐GBD、SSP和CMIP6数据
    ↓ 数据门
统一国家/年龄/性别/年份/情景与空间网格
    ↓
建立1990–2050国家七分类BMI锚点
    ↓
拟合未来变量可获得的MRP
    ↓ 非劣性门 + 时间回测门
生成0.5°主结果、ADM1汇总和0.1°补充结果
    ↓
冻结结直肠癌RR和GBD病因映射
    ↓
计算结直肠癌PAF
    ↓
计算归因发病、死亡和DALY
    ↓
传播不确定性并进行国家层面比较
    ↓ CRC样板门
扩展至其他疾病
```

## 当前停点

第0步已完成；第1步因关键未来数据缺失而停在数据门前。当前正确动作是补齐并冻结机器可读数据，而不是启动模型。
"@
Write-Utf8NoBom -Text $workflow -Path $paths.workflow

$generatedFiles = @(Get-ChildItem -LiteralPath $OutputRoot -Recurse -File -ErrorAction Stop |
    Where-Object FullName -NotLike $paths.sha)
$shaRows = foreach ($file in $generatedFiles) {
    [pscustomobject]@{
        relative_path = [System.IO.Path]::GetRelativePath($OutputRoot, $file.FullName)
        size_bytes = [int64]$file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
Write-Utf8BomCsv -Data $shaRows -Path $paths.sha

[pscustomobject]@{
    status = 'stage00_completed_stage01_blocked_partial'
    output_root = $OutputRoot
    frozen_files = $baselineFileRows.Count
    frozen_bytes = [int64](($baselineFileRows | Measure-Object size_bytes -Sum).Sum)
    inventory_rows = $inventoryRows.Count
    missing_blocking = @($inventoryRows | Where-Object status -eq 'missing_blocking').Count
    next_stage_allowed = $false
} | ConvertTo-Json -Depth 5
